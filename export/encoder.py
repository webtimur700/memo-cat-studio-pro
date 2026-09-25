"""Выбор видеокодера: аппаратный H.264 через VAAPI (Radeon 780M) или программный libx264.

VAAPI на AMD в Fedora работает только с `mesa-va-drivers-freeworld` из RPM Fusion: штатный mesa
собран без H.264/HEVC (scripts/setup.sh ставит freeworld). Наличие `h264_vaapi` в ffmpeg и
даже строки VAEntrypointEncSlice в vainfo ничего не гарантируют, поэтому «работает» проверяется
пробным кодированием крошечного видео (один раз за процесс). Нет устройства/драйвера/профиля —
честно возвращаемся на libx264 (warning в лог), пайплайн не падает.

Качество: у libx264 его задают CRF и пресет, у VAAPI — целевой битрейт как доля потолка из
настроек (см. VAAPI_BITRATE_SHARE): для VAAPI CRF не существует, а VBR с потолком сохраняет
смысл настройки «Битрейт видео (максимум)».
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from core.entities.settings import ExportSettings
from export.quality_presets import QualityPreset

ENCODER_CHOICES = ("auto", "vaapi", "x264")
VAAPI_DEVICE = os.environ.get("MEMO_CAT_VAAPI_DEVICE", "/dev/dri/renderD128")
# Доля потолка битрейта, которую VAAPI берёт целью для пресета качества (замеры: docs/performance.md).
# Без нескольких слайсов mesa (VCN на Radeon 780M) кодирует 1080x1920 с ошибкой в последнем ряду макроблоков: ffmpeg
# при декодировании пишет «error while decoding MB .. 119, bytestream -N» почти на каждом кадре (замерено на 24 клипах,
# `ffmpeg -v error -i clip.mp4 -f null -`). Любое число слайсов >= 2 убирает ошибки без потери скорости и качества.
VAAPI_SLICES = 4
VAAPI_BITRATE_SHARE = {"high": 0.5, "medium": 0.3, "low": 0.15}


@lru_cache(maxsize=1)
def vaapi_h264_works() -> bool:
    """Реально ли получается закодировать H.264 через VAAPI (пробное кодирование 0.2 с)."""
    if not os.path.exists(VAAPI_DEVICE):
        logger.info("VAAPI недоступен: нет устройства {}", VAAPI_DEVICE)
        return False
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-vaapi_device", VAAPI_DEVICE,
        "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=0.2", "-vf", "format=nv12,hwupload",
        "-c:v", "h264_vaapi", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("VAAPI: пробное кодирование не запустилось: {}", exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "VAAPI H.264 не работает ({}) — кодирую через libx264. Нужен mesa-va-drivers-freeworld (RPM Fusion): "
            "`sudo dnf swap mesa-va-drivers mesa-va-drivers-freeworld`",
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"код {result.returncode}",
        )
        return False
    logger.info("VAAPI H.264 работает ({}): аппаратное кодирование клипов", VAAPI_DEVICE)
    return True


def resolve_encoder(preference: str) -> str:
    """'vaapi' или 'x264' по настройке ('auto' — VAAPI, если реально работает)."""
    if preference == "x264":
        return "x264"
    if vaapi_h264_works():
        return "vaapi"
    if preference == "vaapi":
        logger.warning("Кодер VAAPI выбран в настройках, но не работает — использую libx264")
    return "x264"


@dataclass(frozen=True, slots=True)
class EncoderArgs:
    global_args: list[str]     # до первого -i
    filter_suffix: str         # хвост видеофильтра (загрузка кадров на GPU), пусто для x264
    codec_args: list[str]      # параметры кодирования видео


def build_encoder_args(encoder: str, settings: ExportSettings, quality: QualityPreset) -> EncoderArgs:
    """Аргументы ffmpeg для кодера. Потолок битрейта (maxrate) действует у обоих."""
    if encoder == "vaapi":
        share = VAAPI_BITRATE_SHARE.get(settings.quality_preset, VAAPI_BITRATE_SHARE["high"])
        target = max(0.5, settings.bitrate_mbps * share)
        return EncoderArgs(
            ["-vaapi_device", VAAPI_DEVICE],
            "format=nv12,hwupload",
            ["-c:v", "h264_vaapi", "-rc_mode", "VBR", "-b:v", f"{target:.2f}M", "-maxrate", f"{settings.bitrate_mbps}M",
             "-bufsize", f"{2 * settings.bitrate_mbps}M",
             "-slices", str(VAAPI_SLICES)],
        )
    return EncoderArgs(
        [], "",
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", quality.x264_preset, "-crf", str(quality.crf),
         "-maxrate", f"{settings.bitrate_mbps}M", "-bufsize", f"{2 * settings.bitrate_mbps}M"],
    )
