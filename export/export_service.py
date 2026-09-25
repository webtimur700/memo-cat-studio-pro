"""Финальная сборка одного Shorts-ролика (Функция 15: экспорт MP4 H264/AAC
30fps 1080x1920) — объединяет результаты предыдущих Шагов в один вызов
FFmpeg с filter_complex: динамический crop (Шаг 10), burned ASS-субтитры
(Шаг 7), анимированные оверлеи (плашка, логотип, Subscribe; Шаги 9, 13, 14).

Оверлеи рендерятся заранее покадровыми слоями rawvideo RGBA (effects/overlay_layers.py) и
накладываются `overlay` в том же проходе, где видео кодируется, — один проход кодирования
вместо прежних двух и без промежуточного ролика с альфа-каналом. Историческая справка: раньше
это был QuickTime Animation (qtrle) в MOV, потому что libvpx-vp9 в этой сборке ffmpeg молча теряет
альфа-канал (заявленная в `-h` поддержка yuva420p не гарантирует, что она работает — проверять
надо декодированием результата). qtrle-ролик сохранён в прошлом: rawvideo альфу не сжимает вовсе.

Кодер — аппаратный h264_vaapi, если он реально работает, иначе libx264 (export/encoder.py).
Эффекты плагинов требуют покадровой обработки OpenCV, поэтому с ними проход разделён на два:
база (кроп + масштаб) -> эффекты -> субтитры и оверлеи с финальным кодированием.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from core.exceptions import FFmpegExecutionError
from core.stage_timer import stage
from core.entities.settings import ExportSettings
from export.frame_effects import apply_effects_to_video
from effects.branding_overlay import BrandingOverlay
from export.dynamic_crop import CropSample, build_dynamic_crop_filter
from effects.overlay_layers import OverlayLayer, render_overlay_layers
from export.encoder import build_encoder_args, resolve_encoder
from export.quality_presets import resolve_quality_preset


def _run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)
    if result.returncode != 0:
        raise FFmpegExecutionError(command, result.returncode, result.stderr)


@dataclass(frozen=True, slots=True)
class ExportPlan:
    source_path: Path
    output_path: Path
    crop_samples: list[CropSample]
    subtitle_ass_path: Path | None
    banner_rect: tuple[int, int, int, int] | None
    banner_text_lines: list[str]
    banner_appear_at_sec: float
    banner_duration_sec: float
    settings: ExportSettings
    branding: BrandingOverlay | None = None
    """Логотип + кнопка Subscribe для overlay-ролика (None — без них)."""
    effects: tuple = ()
    """Эффекты-плагины: пары (имя, функция кадр BGR -> кадр BGR), выбранные пользователем. Применяются к видеоряду до
    субтитров и оверлеев (логотип и плашка не тонируются). Пусто — дополнительный проход не запускается."""
    source_start_sec: float = 0.0
    """Абсолютное время начала момента в ИСХОДНОМ видео (секунды).

    crop_samples[i].timestamp_sec отсчитывается от начала МОМЕНТА (0-based) —
    без -ss на этот сдвиг ffmpeg декодировал бы всегда с начала исходного
    файла, независимо от того, где реально находится момент. Раньше это
    поле отсутствовало вовсе (баг, найденный по логам реального запуска:
    все клипы падали на экспорте / брали не тот участок видео)."""


def _x264_args(settings: ExportSettings, quality) -> list[str]:
    """Параметры libx264 (CRF/пресет задают качество, maxrate/bufsize — потолок битрейта): проход с эффектами плагинов."""
    return build_encoder_args("x264", settings, quality).codec_args


def _layer_input_args(layer: OverlayLayer, fps: int) -> list[str]:
    """Слой rawvideo RGBA как вход ffmpeg: начинается со своего кадра (-itsoffset), после последнего кадра исчезает."""
    return [
        "-itsoffset", f"{layer.start_frame / fps:.6f}",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-video_size", f"{layer.width}x{layer.height}", "-framerate", str(fps),
        "-i", str(layer.path),
    ]


def _overlay_graph(head_chain: str, layers: list[OverlayLayer], first_layer_input: int, suffix: str) -> tuple[str, str]:
    """filter_complex: [0:v]head_chain[v0], затем наложение слоёв по порядку, затем suffix. Возвращает (граф, метка результата)."""
    parts = [f"[0:v]{head_chain or 'null'}[v0]"]
    for i, layer in enumerate(layers):
        parts.append(
            f"[v{i}][{first_layer_input + i}:v]overlay={layer.x}:{layer.y}:eof_action=pass:format=auto[v{i + 1}]"
        )
    label = f"v{len(layers)}"
    if suffix:
        parts.append(f"[{label}]{suffix}[vout]")
        label = "vout"
    return ";".join(parts), label


class ExportService:
    def __init__(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise FFmpegExecutionError(["ffmpeg"], -1, "ffmpeg не найден в PATH")

    def export_clip(self, plan: ExportPlan) -> Path:
        encoder = resolve_encoder(plan.settings.encoder)
        try:
            return self._export(plan, encoder)
        except FFmpegExecutionError as exc:
            if encoder != "vaapi":
                raise
            # аппаратный кодер может сорваться на конкретном клипе (занят GPU, нехватка памяти): клип важнее скорости
            logger.warning("VAAPI не справился с {} ({}) — повторяю через libx264", plan.output_path.name, str(exc)[:200])
            return self._export(plan, "x264")

    def _export(self, plan: ExportPlan, encoder: str) -> Path:
        quality = resolve_quality_preset(plan.settings.quality_preset)
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        enc = build_encoder_args(encoder, plan.settings, quality)
        fps = plan.settings.fps

        # ffmpeg-фильтрграф — своя мини-грамматика (двоеточия, запятые,
        # квадратные скобки — служебные символы), и экранирование пути внутри
        # неё исторически хрупкое и по-разному ведёт себя на разных сборках
        # ffmpeg. Самый надёжный способ передать путь с субтитрами — вообще
        # не класть его в строку фильтра: запускаем ffmpeg с cwd=папка ass-
        # файла и передаём в фильтр только голое имя файла, без единого
        # спецсимвола пути.
        ass_cwd = plan.subtitle_ass_path.parent if plan.subtitle_ass_path is not None else None
        ass_filter = f"ass={plan.subtitle_ass_path.name}" if plan.subtitle_ass_path is not None else ""
        # с эффектами субтитры вжигаются позже (после эффекта, вместе с оверлеями), иначе эффект тонировал бы и текст
        defer_subtitles = bool(plan.effects) and plan.subtitle_ass_path is not None

        # fps первым: лишние кадры источника (60 fps) отбрасываются до дорогих crop/scale/ass
        head = [f"fps={fps}", build_dynamic_crop_filter(plan.crop_samples), f"scale={plan.settings.width}:{plan.settings.height}"]
        if ass_filter and not defer_subtitles:
            head.append(ass_filter)
        head_chain = ",".join(head)
        clip_duration = plan.crop_samples[-1].timestamp_sec
        has_overlays = plan.banner_rect is not None or (plan.branding is not None and not plan.branding.is_empty)

        with tempfile.TemporaryDirectory(prefix="memo_cat_export_") as tmp_dir:
            tmp = Path(tmp_dir)
            layers: list[OverlayLayer] = []
            if has_overlays:
                with stage("5b экспорт: рендер слоёв overlay"):
                    layers = render_overlay_layers(
                        tmp / "layers",
                        clip_duration_sec=clip_duration,
                        frame_size=(plan.settings.width, plan.settings.height),
                        fps=fps,
                        banner_rect=plan.banner_rect,
                        banner_text_lines=plan.banner_text_lines,
                        appear_at_sec=plan.banner_appear_at_sec,
                        banner_duration_sec=plan.banner_duration_sec,
                        branding=plan.branding,
                    )

            if not plan.effects:
                graph, label = _overlay_graph(head_chain, layers, 1, enc.filter_suffix)
                command = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *enc.global_args,
                    "-ss", f"{plan.source_start_sec:.3f}", "-t", f"{clip_duration:.3f}", "-i", str(plan.source_path.resolve()),
                    *[a for layer in layers for a in _layer_input_args(layer, fps)],
                    "-filter_complex", graph, "-map", f"[{label}]", "-map", "0:a:0?",
                    "-t", f"{clip_duration:.3f}", "-r", str(fps),
                    *enc.codec_args, "-c:a", plan.settings.codec_audio, "-b:a", f"{quality.audio_bitrate_kbps}k",
                    "-movflags", "+faststart", str(plan.output_path.resolve()),
                ]
                logger.info("Экспорт клипа ({}; слоёв overlay: {}): {}", encoder, len(layers), plan.output_path.name)
                logger.debug("ffmpeg filter_complex: {}", graph)
                with stage("5a экспорт: кроп+scale+субтитры+overlay+кодирование"):
                    _run(command, cwd=ass_cwd)
                return plan.output_path

            # --- с эффектами плагинов: база -> эффекты (OpenCV) -> субтитры и оверлеи ---
            base_output = tmp / "base.mp4"
            x264 = build_encoder_args("x264", plan.settings, quality)
            base_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{plan.source_start_sec:.3f}", "-t", f"{clip_duration:.3f}", "-i", str(plan.source_path.resolve()),
                "-vf", head_chain, "-r", str(fps), *x264.codec_args,
                "-c:a", plan.settings.codec_audio, "-b:a", f"{quality.audio_bitrate_kbps}k",
                "-movflags", "+faststart", str(base_output.resolve()),
            ]
            logger.info("Экспорт клипа (база для эффектов): {}", plan.output_path.name)
            with stage("5a экспорт: база для эффектов"):
                _run(base_cmd, cwd=ass_cwd)

            fx_output = tmp / "fx.mp4"
            logger.info("Эффекты плагинов ({}) для клипа {}", ", ".join(n for n, _ in plan.effects), plan.output_path.name)
            with stage("5d экспорт: эффекты плагинов"):
                apply_effects_to_video(base_output, fx_output, plan.effects, fps, x264.codec_args)

            graph, label = _overlay_graph(ass_filter if defer_subtitles else "", layers, 1, enc.filter_suffix)
            final_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *enc.global_args,
                "-i", str(fx_output.resolve()), *[a for layer in layers for a in _layer_input_args(layer, fps)],
                "-filter_complex", graph, "-map", f"[{label}]", "-map", "0:a:0?",
                "-r", str(fps), *enc.codec_args, "-c:a", "copy", "-movflags", "+faststart", str(plan.output_path.resolve()),
            ]
            logger.info("Наложение субтитров/оверлеев на клип с эффектами: {}", plan.output_path.name)
            with stage("5c экспорт: субтитры+overlay+кодирование"):
                _run(final_cmd, cwd=ass_cwd if defer_subtitles else None)
        return plan.output_path
