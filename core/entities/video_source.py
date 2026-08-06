"""Доменная сущность исходного видео.

Заполняется исключительно данными из ffprobe (video/ffmpeg_wrapper.py) —
никаких предположений о длительности/разрешении "по умолчанию", потому что
дальнейший пайплайн (scoring, cutting, reframe) требует точных чисел.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoSource:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    has_audio: bool
    file_size_bytes: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width
