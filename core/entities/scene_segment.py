"""Сегмент сцены — результат работы video/scene_detector.py.

Используется cutting/boundary_refiner.py (Шаг 7), чтобы не резать клип
посреди сцены — границы Shorts всегда выравниваются по ближайшей границе сцены.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SceneSegment:
    index: int
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
