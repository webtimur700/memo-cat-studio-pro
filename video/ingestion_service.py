"""Приём видео (Функция 1): проверка расширения + реальная валидация через
ffprobe (файл с "правильным" расширением, но битым содержимым, будет отклонён
на этапе get_video_source(), а не позже где-нибудь в середине пайплайна).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.entities.video_source import VideoSource
from core.exceptions import UnsupportedFormatError, VideoDecodeError
from video.ffmpeg_wrapper import FFmpegWrapper

SUPPORTED_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

MIN_DURATION_SEC = 3.0   # короче — не наберётся даже на 15-секундный Shorts с запасом
MAX_DURATION_SEC = 4 * 60 * 60  # 4 часа — разумный потолок для одного лонгформ-видео


class IngestionService:
    def __init__(self, ffmpeg: FFmpegWrapper | None = None) -> None:
        self._ffmpeg = ffmpeg or FFmpegWrapper()

    def ingest(self, path: Path) -> VideoSource:
        self._validate_extension(path)
        video_source = self._probe_and_validate(path)
        logger.info(
            "Видео принято: {} ({}x{}, {:.1f}s, {} fps, codec={})",
            path.name,
            video_source.width,
            video_source.height,
            video_source.duration_sec,
            round(video_source.fps, 2),
            video_source.video_codec,
        )
        return video_source

    def ingest_many(self, paths: list[Path]) -> tuple[list[VideoSource], list[tuple[Path, str]]]:
        """Пакетная валидация для Функции 16 (массовая загрузка).

        Не бросает исключение на первом же битом файле — собирает успешные
        VideoSource и список (путь, причина ошибки) для остальных, чтобы
        UI мог показать, какие именно 3 из 100 видео не прошли валидацию.
        """
        accepted: list[VideoSource] = []
        rejected: list[tuple[Path, str]] = []
        for path in paths:
            try:
                accepted.append(self.ingest(path))
            except (UnsupportedFormatError, VideoDecodeError) as exc:
                logger.warning("Видео отклонено: {} — {}", path, exc)
                rejected.append((path, str(exc)))
        return accepted, rejected

    @staticmethod
    def _validate_extension(path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Формат {path.suffix} не поддерживается. "
                f"Разрешены: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def _probe_and_validate(self, path: Path) -> VideoSource:
        video_source = self._ffmpeg.get_video_source(path)

        if video_source.duration_sec < MIN_DURATION_SEC:
            raise VideoDecodeError(
                f"Видео {path.name} слишком короткое ({video_source.duration_sec:.1f}s), "
                f"минимум {MIN_DURATION_SEC:.0f}s."
            )
        if video_source.duration_sec > MAX_DURATION_SEC:
            raise VideoDecodeError(
                f"Видео {path.name} слишком длинное ({video_source.duration_sec / 60:.0f} мин), "
                f"максимум {MAX_DURATION_SEC / 3600:.0f} ч."
            )
        if video_source.width == 0 or video_source.height == 0:
            raise VideoDecodeError(f"Не удалось определить разрешение видео {path.name}.")

        return video_source
