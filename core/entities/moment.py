"""Момент — результат анализа+скоринга (Функции 2-3), кандидат на нарезку."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.entities.detection import Detection
from core.entities.score import SignalPart


@dataclass(frozen=True, slots=True)
class Moment:
    start_sec: float
    end_sec: float
    viral_score: int  # 0-100
    motion_intensity: float = 0.0
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    label: str = ""
    breakdown: tuple[SignalPart, ...] = field(default_factory=tuple)   # вклад сигналов в viral_score (пусто — не считалось)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
