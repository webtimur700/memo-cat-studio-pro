"""Выбор финальных моментов для нарезки (Функция 4).

Вход — оценённые скользящие окна по всему видео (WindowScore), выход —
непересекающиеся Moment, расширенные до ближайшей разрешённой длительности
Shorts (config: shorts.allowed_durations_sec), отсортированные по времени.

Жадный алгоритм: сортируем окна по score по убыванию, берём окно, если оно
не пересекается с уже выбранными — простой и предсказуемый выбор.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.detection import Detection
from core.entities.moment import Moment
from core.entities.settings import UserSettings


@dataclass(frozen=True, slots=True)
class WindowScore:
    start_sec: float
    end_sec: float
    viral_score: int
    motion_intensity: float
    detections: tuple[Detection, ...] = ()


def _expand_to_allowed_duration(
    window: WindowScore, allowed_durations: tuple[int, ...], video_duration_sec: float
) -> tuple[float, float]:
    window_len = window.end_sec - window.start_sec
    target_duration = min((d for d in allowed_durations if d >= window_len), default=max(allowed_durations))
    target_duration = min(target_duration, video_duration_sec)

    center = (window.start_sec + window.end_sec) / 2
    half = target_duration / 2

    start = max(0.0, center - half)
    end = min(video_duration_sec, start + target_duration)
    start = max(0.0, end - target_duration)

    return start, end


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def select_moments(
    window_scores: list[WindowScore],
    settings: UserSettings,
    video_duration_sec: float,
    max_moments: int = 10,
) -> list[Moment]:
    threshold = settings.viral_score.queue_threshold
    candidates = [w for w in window_scores if w.viral_score >= threshold]
    candidates.sort(key=lambda w: w.viral_score, reverse=True)

    selected_ranges: list[tuple[float, float]] = []
    moments: list[Moment] = []

    for window in candidates:
        if len(moments) >= max_moments:
            break

        start, end = _expand_to_allowed_duration(
            window, settings.shorts.allowed_durations_sec, video_duration_sec
        )
        candidate_range = (start, end)

        if any(_overlaps(candidate_range, existing) for existing in selected_ranges):
            continue

        selected_ranges.append(candidate_range)
        moments.append(
            Moment(
                start_sec=start,
                end_sec=end,
                viral_score=window.viral_score,
                motion_intensity=window.motion_intensity,
                detections=window.detections,
            )
        )

    moments.sort(key=lambda m: m.start_sec)
    return moments
