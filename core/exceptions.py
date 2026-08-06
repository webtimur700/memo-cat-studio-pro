"""Доменные исключения. Живут в core, потому что и Infrastructure (video/, vision/),
и Application-сервисы должны уметь их ловить, не зная деталей реализации друг друга.
"""

from __future__ import annotations


class MemoCatError(Exception):
    """Базовое исключение приложения."""


class VideoDecodeError(MemoCatError):
    """Файл не читается как видео: повреждён, неизвестный кодек, нулевая длительность и т.п."""


class UnsupportedFormatError(MemoCatError):
    """Расширение файла не входит в поддерживаемый список (mp4/mov/avi/mkv/webm)."""


class FFmpegExecutionError(MemoCatError):
    """FFmpeg/FFprobe завершился с ненулевым кодом возврата."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Команда {' '.join(command)} завершилась с кодом {returncode}: {stderr.strip()[-500:]}"
        )


class SceneDetectionError(MemoCatError):
    """Ошибка в процессе детекции сцен (PySceneDetect)."""
