"""Финальная сборка одного Shorts-ролика (Функция 15: экспорт MP4 H264/AAC
30fps 1080x1920) — объединяет результаты предыдущих Шагов в один вызов
FFmpeg с filter_complex: динамический crop (Шаг 10), burned ASS-субтитры
(Шаг 7), overlay анимированной плашки как отдельного видео с альфа-каналом
(Шаг 9).

Оверлей плашки рендерится заранее в отдельный MOV с альфа-каналом (QuickTime
Animation / qtrle) через render_banner_overlay_clip() — рендерить анимацию
покадрово внутри одного gigantic filter_complex было бы на порядок сложнее
и хрупче, чем скомпоновать готовый прозрачный ролик и наложить его один раз
через `overlay`.

ЧЕСТНОЕ ПРИМЕЧАНИЕ (реальная проблема, пойманная тестом на этом шаге):
изначально для промежуточного прозрачного ролика использовался WebM/VP9
(`-pix_fmt yuva420p`) — стандартный с виду выбор для "видео с альфа-каналом".
На практике в этой сборке ffmpeg (и, вероятно, во многих других без
специальных патчей) libvpx-vp9 заявляет поддержку yuva420p в `-h
encoder=libvpx-vp9`, но реально MOLча теряет альфа-канал при кодировании —
задокументированная в `-h` поддержка формата не гарантирует, что она
реально работает в контейнере, это стоит перепроверять декодированием
результата обратно, а не только чтением списка поддерживаемых pix_fmt.
QuickTime Animation (`qtrle`) в MOV — старый, но надёжно сохраняет альфа-канал
и на этом ffmpeg подтверждён реальным round-trip тестом (кодирование ->
декодирование -> проверка альфа-пикселей).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image

from animation.promo_banner_animator import compute_banner_animation_state, render_banner_on_frame
from core.exceptions import FFmpegExecutionError
from core.entities.settings import ExportSettings
from export.dynamic_crop import CropSample, build_dynamic_crop_filter
from export.quality_presets import resolve_quality_preset


def _run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)
    if result.returncode != 0:
        raise FFmpegExecutionError(command, result.returncode, result.stderr)


def render_banner_overlay_clip(
    output_path: Path,
    clip_duration_sec: float,
    banner_rect: tuple[int, int, int, int],
    banner_text_lines: list[str],
    appear_at_sec: float,
    banner_duration_sec: float,
    frame_size: tuple[int, int],
    fps: int = 30,
) -> Path:
    """Рендерит ТОЛЬКО анимированную плашку (прозрачный фон) на весь клип —
    PNG-последовательность через Pillow, затем кодируется в MOV/qtrle с
    реально сохранённым альфа-каналом (см. честное примечание в docstring
    модуля про WebM/VP9). output_path должен иметь расширение .mov.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size

    with tempfile.TemporaryDirectory(prefix="memo_cat_banner_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        total_frames = max(1, int(clip_duration_sec * fps))

        transparent_frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        for frame_index in range(total_frames):
            t = frame_index / fps
            elapsed_since_appear = t - appear_at_sec
            state = compute_banner_animation_state(elapsed_since_appear, banner_duration_sec)

            if state.visible and state.opacity > 0.01:
                frame_img = render_banner_on_frame(
                    transparent_frame, state, banner_rect, banner_text_lines
                )
            else:
                frame_img = transparent_frame

            frame_img.save(tmp_path / f"frame_{frame_index:06d}.png")

        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(tmp_path / "frame_%06d.png"),
            "-c:v", "qtrle",
            str(output_path),
        ])

    return output_path


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
    source_start_sec: float = 0.0
    """Абсолютное время начала момента в ИСХОДНОМ видео (секунды).

    crop_samples[i].timestamp_sec отсчитывается от начала МОМЕНТА (0-based) —
    без -ss на этот сдвиг ffmpeg декодировал бы всегда с начала исходного
    файла, независимо от того, где реально находится момент. Раньше это
    поле отсутствовало вовсе (баг, найденный по логам реального запуска:
    все клипы падали на экспорте / брали не тот участок видео)."""


class ExportService:
    def __init__(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise FFmpegExecutionError(["ffmpeg"], -1, "ffmpeg не найден в PATH")

    def export_clip(self, plan: ExportPlan) -> Path:
        quality = resolve_quality_preset(plan.settings.quality_preset)
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)

        crop_filter = build_dynamic_crop_filter(plan.crop_samples)
        scale_filter = f"scale={plan.settings.width}:{plan.settings.height}"
        video_filters = [crop_filter, scale_filter]

        # ffmpeg-фильтрграф — своя мини-грамматика (двоеточия, запятые,
        # квадратные скобки — служебные символы), и экранирование пути внутри
        # неё исторически хрупкое и по-разному ведёт себя на разных сборках
        # ffmpeg. Самый надёжный способ передать путь с субтитрами — вообще
        # не класть его в строку фильтра: запускаем ffmpeg с cwd=папка ass-
        # файла и передаём в фильтр только голое имя файла, без единого
        # спецсимвола пути.
        ass_cwd: Path | None = None
        if plan.subtitle_ass_path is not None:
            ass_cwd = plan.subtitle_ass_path.parent
            video_filters.append(f"ass={plan.subtitle_ass_path.name}")

        filter_chain = ",".join(video_filters)

        with tempfile.TemporaryDirectory(prefix="memo_cat_export_") as tmp_dir:
            base_output = Path(tmp_dir) / "base.mp4"
            clip_duration = plan.crop_samples[-1].timestamp_sec

            base_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{plan.source_start_sec:.3f}",
                "-i", str(plan.source_path.resolve()),
                "-t", f"{clip_duration:.3f}",
                "-vf", filter_chain,
                "-r", str(plan.settings.fps),
                "-c:v", "libx264",
                "-preset", quality.x264_preset,
                "-crf", str(quality.crf),
                "-c:a", plan.settings.codec_audio,
                "-b:a", f"{quality.audio_bitrate_kbps}k",
                "-movflags", "+faststart",
                str(base_output.resolve()),
            ]
            logger.info("Экспорт клипа (база: crop+scale+субтитры): {}", plan.output_path.name)
            logger.debug("ffmpeg filter_chain: {}", filter_chain)
            _run(base_cmd, cwd=ass_cwd)

            if plan.banner_rect is None:
                shutil.copy(base_output, plan.output_path)
                return plan.output_path

            banner_overlay_path = Path(tmp_dir) / "banner_overlay.mov"
            render_banner_overlay_clip(
                banner_overlay_path,
                clip_duration_sec=clip_duration,
                banner_rect=plan.banner_rect,
                banner_text_lines=plan.banner_text_lines,
                appear_at_sec=plan.banner_appear_at_sec,
                banner_duration_sec=plan.banner_duration_sec,
                frame_size=(plan.settings.width, plan.settings.height),
                fps=plan.settings.fps,
            )

            logger.info("Наложение анимированной плашки на клип: {}", plan.output_path.name)
            overlay_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(base_output),
                "-i", str(banner_overlay_path),
                "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
                "-c:a", "copy",
                str(plan.output_path),
            ]
            _run(overlay_cmd)

        return plan.output_path
