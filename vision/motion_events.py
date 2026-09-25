"""Прыжки, падения и рывки в окне видео для Viral Score (vision/pose_motion_analyzer.py по трекам YOLO).

Анализатору нужны треки с шагом 0.1-0.2 с (окно разворота направления — 0.6 с), а основной скан идёт на 1 кадре/с,
поэтому для лучших окон снимается плотный трек: кадры читаются последовательно на MOTION_SAMPLE_FPS, животные
детектируются YOLO и собираются в трек ObjectTracker'ом, главный трек уходит в PoseMotionAnalyzer.

Ложные события на монтажных склейках (bbox «прыгает» между разными сценами) отбрасываются: событие в пределах
CUT_GUARD_SEC от смены сцены не считается.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.entities.detection import is_animal_class
from video.frame_extractor import FrameExtractor
from vision.parallel_detect import detected_stream
from vision.pose_motion_analyzer import MotionEvent, MotionEventType, PoseMotionAnalyzer
from vision.tracker import ObjectTracker

MOTION_SAMPLE_FPS = 6.0
CUT_GUARD_SEC = 0.4
JUMP_FALL_POINTS = 0.5        # два прыжка/падения — сигнал на максимум
RAPID_POINTS = 0.1            # рывки частые (камера трясётся), поэтому дёшево и не больше RAPID_MAX_POINTS
RAPID_MAX_POINTS = 0.3


@dataclass(frozen=True, slots=True)
class MotionSummary:
    jumps: int = 0
    falls: int = 0
    rapid: int = 0
    value: float = 0.0            # 0..1 для ScoreInputs.motion_events

    @property
    def detail(self) -> str:
        parts = []
        if self.jumps:
            parts.append(f"{self.jumps} {_plural(self.jumps, 'прыжок', 'прыжка', 'прыжков')}")
        if self.falls:
            parts.append(f"{self.falls} {_plural(self.falls, 'падение', 'падения', 'падений')}")
        if self.rapid:
            parts.append(f"{self.rapid} {_plural(self.rapid, 'рывок', 'рывка', 'рывков')}")
        return ", ".join(parts)


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def summarize_events(events: list[MotionEvent], scene_cuts: list[float] = ()) -> MotionSummary:
    """События окна -> счётчики и значение 0..1; события у монтажных склеек отбрасываются."""
    def near_cut(t: float) -> bool:
        return any(abs(t - cut) <= CUT_GUARD_SEC for cut in scene_cuts)

    kept = [e for e in events if not near_cut(e.timestamp_sec)]
    jumps = sum(1 for e in kept if e.event_type == MotionEventType.JUMP)
    falls = sum(1 for e in kept if e.event_type == MotionEventType.FALL)
    rapid = sum(1 for e in kept if e.event_type == MotionEventType.RAPID_MOVEMENT)
    value = min(1.0, (jumps + falls) * JUMP_FALL_POINTS + min(RAPID_MAX_POINTS, rapid * RAPID_POINTS))
    return MotionSummary(jumps, falls, rapid, value)


def analyze_window_motion(
    extractor: FrameExtractor, detector, start_sec: float, end_sec: float, scene_cuts: list[float] = ()
) -> MotionSummary:
    """Плотный трек животного в окне [start, end) и события движения по нему."""
    tracker = ObjectTracker()
    with detected_stream(detector, extractor.dense_frames_in_range(start_sec, end_sec, MOTION_SAMPLE_FPS)) as stream:
        for timestamp, _frame, detections in stream:
            tracker.update([replace(d, frame_timestamp_sec=timestamp) for d in detections if is_animal_class(d.class_id)])
    primary = tracker.primary_track()
    if primary is None or len(primary.detections) < 3:
        return MotionSummary()
    result = PoseMotionAnalyzer().analyze(primary.detections)
    return summarize_events(result.events, [c for c in scene_cuts if start_sec - 1 <= c <= end_sec + 1])
