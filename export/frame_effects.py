"""Применение эффектов-плагинов (PluginRegistry.get_effect) к готовому видеоряду клипа.

Эффект — функция кадр BGR (numpy uint8, H×W×3) -> кадр BGR. Кадры читаются OpenCV, проходят выбранные эффекты по порядку
и уходят в ffmpeg по трубе (звук берётся из исходного файла без перекодирования). Работает только когда пользователь выбрал
эффекты: без них этот проход не запускается вовсе, экспорт не замедляется.

Плагин — чужой код: эффект, бросивший исключение или вернувший кадр другой формы, отключается для этого клипа (в лог),
остальные эффекты и клип продолжают работать.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from loguru import logger

from core.exceptions import FFmpegExecutionError

Effect = Callable[[np.ndarray], np.ndarray]


def apply_effects_to_video(
    source: Path,
    output: Path,
    effects: Sequence[tuple[str, Effect]],
    fps: float,
    video_args: list[str],
) -> list[str]:
    """Пишет в output видео source с применёнными effects (пары «имя, функция»), звук копируется.
    video_args — параметры кодирования видео (CRF/пресет/maxrate). Возвращает имена реально применённых эффектов."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FFmpegExecutionError(["cv2.VideoCapture", str(source)], -1, "не удалось открыть видео для эффектов")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{source_fps}", "-i", "pipe:0",
        "-i", str(source), "-map", "0:v", "-map", "1:a?",
        "-pix_fmt", "yuv420p", *video_args, "-c:a", "copy", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    active = list(effects)
    failed: set[str] = set()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for name, effect in list(active):
                try:
                    result = effect(frame)
                    if result.shape != frame.shape or result.dtype != np.uint8:
                        raise ValueError(f"эффект вернул кадр {result.shape}/{result.dtype}, ожидалось {frame.shape}/uint8")
                    frame = result
                except Exception as exc:
                    logger.warning("Эффект '{}' отключён для этого клипа: {}", name, exc)
                    active.remove((name, effect))
                    failed.add(name)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise FFmpegExecutionError(command, process.returncode, stderr)
    except BrokenPipeError:
        raise FFmpegExecutionError(command, process.wait(), process.stderr.read().decode("utf-8", "replace"))
    finally:
        capture.release()
        if process.poll() is None:
            process.kill()
    return [name for name, _ in effects if name not in failed]
