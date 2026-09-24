"""Готовый нарезанный клип — результат работы пайплайна на один Moment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.entities.moment import Moment


@dataclass(frozen=True, slots=True)
class Clip:
    moment: Moment
    output_path: Path
    title: str = ""
    description: str = ""
    hashtags: tuple[str, ...] = field(default_factory=tuple)
    titles: tuple[str, ...] = field(default_factory=tuple)
    transcript: str = ""
    metadata_path: Path | None = None
    cover_path: Path | None = None
    subtitle_paths: tuple[Path, ...] = field(default_factory=tuple)   # SRT/ASS рядом с клипом, тайминг от начала клипа
