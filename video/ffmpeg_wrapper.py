"""Обёртка над FFmpeg/FFprobe.

Все вызовы — реальный subprocess, без имитации: probe() реально парсит JSON
от ffprobe, extract_segment() реально запускает ffmpeg и возвращает путь к
созданному файлу (или бросает FFmpegExecutionError, если ffmpeg вернул
ненулевой код).

VAAPI-детекция честная: используется только если `ffmpeg -hwaccels` реально
перечисляет vaapi И `/dev/dri/renderD128` существует на диске — на машинах без
GPU (например, эта песочница, или headless-сервер) is_vaapi_available()
корректно вернёт False, а не соврёт "да".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.entities.video_source import VideoSource
from core.exceptions import FFmpegExecutionError, VideoDecodeError

DEFAULT_VAAPI_DEVICE = Path("/dev/dri/renderD128")


@dataclass(frozen=True, slots=True)
class TranscodeOptions:
    fps: int = 30
    width: int = 1080
    height: int = 1920
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    bitrate_mbps: int = 12
    use_hwaccel: bool = False


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegExecutionError(command, result.returncode, result.stderr)
    return result


@lru_cache(maxsize=1)
def is_vaapi_available() -> bool:
    """Реальная проверка, а не предположение по ОС. Кэшируется на процесс —
    драйверы не появляются/исчезают в рантайме.
    """
    if shutil.which("ffmpeg") is None:
        return False
    if not DEFAULT_VAAPI_DEVICE.exists():
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return "vaapi" in result.stdout.lower()


class FFmpegWrapper:
    def __init__(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise VideoDecodeError(
                "ffmpeg/ffprobe не найдены в PATH — установите пакет ffmpeg."
            )

    def probe(self, path: Path) -> dict:
        if not path.exists():
            raise VideoDecodeError(f"Файл не найден: {path}")

        command = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = _run(command)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise VideoDecodeError(f"ffprobe вернул невалидный JSON для {path}") from exc

    def get_video_source(self, path: Path) -> VideoSource:
        probe_data = self.probe(path)

        video_stream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None
        )
        format_info = probe_data.get("format", {})

        if video_stream is None:
            raise VideoDecodeError(f"В файле {path} не найдена видеодорожка.")

        duration_raw = format_info.get("duration") or video_stream.get("duration")
        if duration_raw is None:
            raise VideoDecodeError(f"Не удалось определить длительность видео {path}.")

        fps = self._parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))

        return VideoSource(
            path=path,
            duration_sec=float(duration_raw),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            has_audio=audio_stream is not None,
            file_size_bytes=int(format_info.get("size", 0)),
        )

    @staticmethod
    def _parse_frame_rate(raw: str) -> float:
        try:
            numerator, denominator = raw.split("/")
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    def extract_segment(
        self,
        source: Path,
        start_sec: float,
        end_sec: float,
        destination: Path,
        options: TranscodeOptions | None = None,
    ) -> Path:
        """Вырезает и перекодирует сегмент [start_sec, end_sec) в 9:16 MP4.

        Использует crop+scale под целевое разрешение с сохранением пропорций
        через паддинг (letterbox чёрными полосами) — реальное AI Smart Crop
        (по треку животного) применяется ДО этого вызова в vision/smart_crop.py
        (Шаг 6), здесь только гарантированное безопасное приведение к 1080x1920,
        если кадрирование по какой-то причине не сработало.
        """
        options = options or TranscodeOptions()
        destination.parent.mkdir(parents=True, exist_ok=True)

        use_hwaccel = options.use_hwaccel and is_vaapi_available()

        scale_filter = (
            f"scale={options.width}:{options.height}:force_original_aspect_ratio=decrease,"
            f"pad={options.width}:{options.height}:(ow-iw)/2:(oh-ih)/2"
        )

        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if use_hwaccel:
            command += ["-hwaccel", "vaapi", "-hwaccel_device", str(DEFAULT_VAAPI_DEVICE)]

        command += [
            "-ss", f"{start_sec:.3f}",
            "-to", f"{end_sec:.3f}",
            "-i", str(source),
            "-vf", scale_filter,
            "-r", str(options.fps),
            "-c:v", options.video_codec,
            "-b:v", f"{options.bitrate_mbps}M",
            "-c:a", options.audio_codec,
            "-movflags", "+faststart",
            str(destination),
        ]

        _run(command)
        return destination

    def extract_audio_track(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # частота, ожидаемая Whisper/Silero VAD
            "-ac", "1",
            str(destination),
        ]
        _run(command)
        return destination
