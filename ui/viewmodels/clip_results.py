"""Результат обработки одного клипа для интерфейса: клип, обложка, Viral Score,
заголовки, описание, хештеги. Строится из Clip (сразу после пайплайна) или из
<клип>.json в папке экспорта (результаты прошлых запусков не пропадают)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from core.entities.clip import Clip


@dataclass(frozen=True, slots=True)
class ClipResult:
    clip_path: Path
    source_video: str
    start_sec: float
    end_sec: float
    viral_score: int
    title: str
    cover_path: Path | None = None
    titles: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    hashtags: tuple[str, ...] = field(default_factory=tuple)
    transcript: str = ""

    @property
    def folder(self) -> Path:
        return self.clip_path.parent

    @property
    def hashtags_text(self) -> str:
        return " ".join(self.hashtags)

    @property
    def time_range_text(self) -> str:
        def fmt(sec: float) -> str:
            minutes, seconds = divmod(int(sec), 60)
            return f"{minutes:02d}:{seconds:02d}"

        return f"{fmt(self.start_sec)}–{fmt(self.end_sec)}"

    def full_text(self) -> str:
        """Всё для публикации одним блоком: заголовки, описание, хештеги."""
        parts = []
        if self.titles:
            parts.append("Заголовки:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(self.titles, 1)))
        elif self.title:
            parts.append(f"Заголовок: {self.title}")
        if self.description:
            parts.append(f"Описание:\n{self.description}")
        if self.hashtags:
            parts.append(f"Хештеги:\n{self.hashtags_text}")
        return "\n\n".join(parts)

    @classmethod
    def from_clip(cls, clip: Clip, source_video: str) -> "ClipResult":
        return cls(
            clip_path=clip.output_path,
            source_video=source_video,
            start_sec=clip.moment.start_sec,
            end_sec=clip.moment.end_sec,
            viral_score=clip.moment.viral_score,
            title=clip.title,
            cover_path=clip.cover_path,
            titles=tuple(clip.titles),
            description=clip.description,
            hashtags=tuple(clip.hashtags),
            transcript=clip.transcript,
        )

    @classmethod
    def from_json(cls, json_path: Path) -> "ClipResult | None":
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            clip_path = json_path.with_name(data["clip_file"])
            if not clip_path.is_file():
                return None
            cover_name = data.get("cover_file")
            cover_path = json_path.with_name(cover_name) if cover_name else None
            return cls(
                clip_path=clip_path,
                source_video=data.get("source_video", ""),
                start_sec=float(data.get("start_sec", 0.0)),
                end_sec=float(data.get("end_sec", 0.0)),
                viral_score=int(data.get("viral_score", 0)),
                title=data.get("title", ""),
                cover_path=cover_path if cover_path and cover_path.is_file() else None,
                titles=tuple(data.get("titles", ())),
                description=data.get("description", ""),
                hashtags=tuple(data.get("hashtags", ())),
                transcript=data.get("transcript", ""),
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Не удалось прочитать результат {}: {}", json_path.name, exc)
            return None


def load_results_from_dir(output_dir: Path) -> list[ClipResult]:
    """Все результаты из папки экспорта: новые сверху."""
    if not output_dir.is_dir():
        return []
    files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = [ClipResult.from_json(path) for path in files]
    return [r for r in results if r is not None]
