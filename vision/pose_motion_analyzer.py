"""Эвристический анализ движения по треку (Функция 2: "прыжки", "падения",
"быстрое движение", "неожиданная реакция").

ЧЕСТНО (как проговорено в архитектуре, Шаг 1): это НЕ распознавание эмоций
животного нейросетью — такой готовой общедоступной модели не существует.
Это анализ траектории bbox из vision/tracker.py: скорость и направление
движения центра bbox, нормализованные на размер объекта (диагональ bbox),
чтобы результат не зависел от того, насколько животное близко к камере.

Эвристики:
  - motion_intensity: сглаженная скорость движения центра, 0..1
  - JUMP: резкое движение центра вверх (уменьшение y), затем резкое вниз,
    в пределах короткого окна — характерная парабола прыжка
  - FALL: резкое движение вниз БЕЗ предшествующего движения вверх, за которым
    следует резкая остановка (падение = "рывок вниз + импакт")
  - RAPID_MOVEMENT: скорость выше порога безотносительно направления
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.entities.detection import Detection


class MotionEventType(str, Enum):
    JUMP = "jump"
    FALL = "fall"
    RAPID_MOVEMENT = "rapid_movement"


@dataclass(frozen=True, slots=True)
class MotionEvent:
    event_type: MotionEventType
    timestamp_sec: float
    magnitude: float  # относительная величина эффекта, 0..1+, для сортировки по значимости


@dataclass(frozen=True, slots=True)
class MotionAnalysisResult:
    motion_intensity: float  # 0..1, сглаженная средняя скорость по всему треку
    events: list[MotionEvent]


# Пороговые значения подобраны в единицах "доли диагонали bbox за секунду" —
# то есть независимо от разрешения видео и удалённости животного от камеры.
JUMP_VERTICAL_THRESHOLD = 0.8      # доля диагонали/сек
FALL_VERTICAL_THRESHOLD = 0.9
RAPID_MOVEMENT_THRESHOLD = 1.2
DIRECTION_REVERSAL_WINDOW_SEC = 0.6


def _center_y_velocity(prev: Detection, curr: Detection) -> float:
    dt = curr.frame_timestamp_sec - prev.frame_timestamp_sec
    if dt <= 0:
        return 0.0
    _, prev_cy = prev.bbox.center
    _, curr_cy = curr.bbox.center
    diagonal = math.hypot(curr.bbox.width, curr.bbox.height) or 1.0
    return (curr_cy - prev_cy) / diagonal / dt  # положительное = движение вниз (Y растёт вниз)


def _speed(prev: Detection, curr: Detection) -> float:
    dt = curr.frame_timestamp_sec - prev.frame_timestamp_sec
    if dt <= 0:
        return 0.0
    prev_cx, prev_cy = prev.bbox.center
    curr_cx, curr_cy = curr.bbox.center
    diagonal = math.hypot(curr.bbox.width, curr.bbox.height) or 1.0
    displacement = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
    return displacement / diagonal / dt


class PoseMotionAnalyzer:
    def analyze(self, detections: list[Detection]) -> MotionAnalysisResult:
        if len(detections) < 2:
            return MotionAnalysisResult(motion_intensity=0.0, events=[])

        ordered = sorted(detections, key=lambda d: d.frame_timestamp_sec)

        vertical_velocities: list[tuple[float, float]] = []  # (timestamp, v_y)
        speeds: list[float] = []

        for prev, curr in zip(ordered, ordered[1:]):
            vertical_velocities.append((curr.frame_timestamp_sec, _center_y_velocity(prev, curr)))
            speeds.append(_speed(prev, curr))

        motion_intensity = min(1.0, sum(speeds) / len(speeds) / RAPID_MOVEMENT_THRESHOLD)

        events: list[MotionEvent] = []
        events.extend(self._detect_jumps(vertical_velocities))
        events.extend(self._detect_falls(vertical_velocities))
        events.extend(self._detect_rapid_movement(ordered, speeds))

        return MotionAnalysisResult(motion_intensity=motion_intensity, events=events)

    @staticmethod
    def _detect_jumps(vertical_velocities: list[tuple[float, float]]) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        for i in range(1, len(vertical_velocities)):
            t_prev, v_prev = vertical_velocities[i - 1]
            t_curr, v_curr = vertical_velocities[i]

            # Прыжок: движение вверх (v < 0) сменяется движением вниз (v > 0)
            # в пределах короткого окна — характерная "парабола" прыжка.
            if (
                v_prev < -JUMP_VERTICAL_THRESHOLD
                and v_curr > JUMP_VERTICAL_THRESHOLD
                and (t_curr - t_prev) <= DIRECTION_REVERSAL_WINDOW_SEC
            ):
                magnitude = (abs(v_prev) + abs(v_curr)) / (2 * JUMP_VERTICAL_THRESHOLD)
                events.append(MotionEvent(MotionEventType.JUMP, t_curr, magnitude))
        return events

    @staticmethod
    def _detect_falls(vertical_velocities: list[tuple[float, float]]) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        for i in range(1, len(vertical_velocities)):
            t_prev, v_prev = vertical_velocities[i - 1]
            t_curr, v_curr = vertical_velocities[i]

            # Падение: резкое движение вниз, за которым следует резкая остановка
            # (|v_curr| падает почти до нуля) — "рывок вниз + импакт об землю".
            if (
                v_prev > FALL_VERTICAL_THRESHOLD
                and abs(v_curr) < FALL_VERTICAL_THRESHOLD * 0.25
                and (t_curr - t_prev) <= DIRECTION_REVERSAL_WINDOW_SEC
            ):
                events.append(MotionEvent(MotionEventType.FALL, t_curr, v_prev / FALL_VERTICAL_THRESHOLD))
        return events

    @staticmethod
    def _detect_rapid_movement(ordered: list[Detection], speeds: list[float]) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        for detection, speed in zip(ordered[1:], speeds):
            if speed > RAPID_MOVEMENT_THRESHOLD:
                events.append(
                    MotionEvent(
                        MotionEventType.RAPID_MOVEMENT,
                        detection.frame_timestamp_sec,
                        speed / RAPID_MOVEMENT_THRESHOLD,
                    )
                )
        return events
