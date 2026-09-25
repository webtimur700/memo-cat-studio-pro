from pathlib import Path

import cv2
import numpy as np
import pytest

from core.entities.detection import BoundingBox, Detection
from core.entities.settings import UserSettings, ViralScoreSettings
from cutting.clip_selector_service import WindowScore
from pipeline.pipeline_runner import PipelineRunner, _DetectorHandle
from scoring.viral_score_service import ScoreInputs, compute_score_breakdown, compute_viral_score
from video.frame_extractor import FrameExtractor
from vision.motion_events import analyze_window_motion, summarize_events
from vision.pose_motion_analyzer import MotionEvent, MotionEventType

FPS = 30
CAT = 15


class _BlobDetector:
    """Вместо YOLO: ищет яркий квадрат и отдаёт его как кошку."""

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 200)
        if xs.size == 0:
            return []
        return [Detection(CAT, "cat", 0.9, BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))]


def _make_video(path: Path, seconds: float, y_of_t) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for i in range(int(seconds * FPS)):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        y = int(y_of_t(i / FPS))
        cv2.rectangle(frame, (140, y), (180, y + 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _jumping(t):        # покой, на одном плотном кадре (1.1-1.25 с) животное подскакивает на 20 px (полкорпуса) и возвращается
    return 150 if 1.1 <= t < 1.25 else 170


def _still(t):
    return 100


def test_summarize_counts_caps_value_and_drops_events_at_scene_cuts():
    events = [MotionEvent(MotionEventType.JUMP, 2.0, 1.5), MotionEvent(MotionEventType.FALL, 3.0, 1.2),
              MotionEvent(MotionEventType.RAPID_MOVEMENT, 3.5, 1.0), MotionEvent(MotionEventType.JUMP, 4.0, 1.1)]
    summary = summarize_events(events)
    assert (summary.jumps, summary.falls, summary.rapid) == (2, 1, 1)
    assert summary.value == 1.0 and summary.detail == "2 прыжка, 1 падение, 1 рывок"
    cut = summarize_events(events, scene_cuts=[3.2, 3.9])           # склейки рядом с падением, рывком и вторым прыжком
    assert (cut.jumps, cut.falls, cut.rapid) == (1, 0, 0) and cut.value == 0.5
    many_rapid = summarize_events([MotionEvent(MotionEventType.RAPID_MOVEMENT, t, 1.0) for t in range(10)])
    assert many_rapid.value == pytest.approx(0.3)                    # рывки частые — потолок вклада
    assert summarize_events([]).value == 0.0 and summarize_events([]).detail == ""


def test_plural_forms():
    from vision.motion_events import MotionSummary

    assert MotionSummary(jumps=1).detail == "1 прыжок" and MotionSummary(jumps=5).detail == "5 прыжков"
    assert MotionSummary(jumps=11).detail == "11 прыжков" and MotionSummary(jumps=22).detail == "22 прыжка"


def test_dense_frames_are_sequential_at_the_requested_rate(tmp_path):
    video = _make_video(tmp_path / "v.mp4", 3.0, _still)
    with FrameExtractor(video) as extractor:
        stamps = [t for t, _ in extractor.dense_frames_in_range(0.5, 2.5, 6.0)]
    assert len(stamps) in (12, 13) and stamps[0] == pytest.approx(0.5)
    assert all(0.15 < b - a < 0.2 for a, b in zip(stamps, stamps[1:]))


def test_jump_is_found_by_dense_track_and_a_still_animal_gives_nothing(tmp_path):
    jump = _make_video(tmp_path / "jump.mp4", 3.0, _jumping)
    still = _make_video(tmp_path / "still.mp4", 3.0, _still)
    with FrameExtractor(jump) as ex:
        found = analyze_window_motion(ex, _BlobDetector(), 0.0, 3.0)
    with FrameExtractor(still) as ex:
        nothing = analyze_window_motion(ex, _BlobDetector(), 0.0, 3.0)
    assert found.jumps >= 1 and found.value >= 0.5
    assert nothing.value == 0.0 and nothing.detail == ""


def test_breakdown_sums_to_score_and_motion_events_is_a_pure_bonus():
    weights = ViralScoreSettings()
    base_inputs = ScoreInputs(0.4, 0.5, 1, audio_event=0.8, audio_detail="лай")
    base = compute_score_breakdown(base_inputs, weights)
    assert base.score == compute_viral_score(base_inputs, weights)
    assert sum(p.points for p in base.parts) == pytest.approx(base.score, abs=0.5)
    assert {p.key for p in base.parts} == {"motion", "presence", "scene", "audio", "motion_events"}
    assert base.part("audio").detail == "лай" and base.part("motion_events").points == 0 and base.part("motion_events").bonus

    boosted = compute_score_breakdown(ScoreInputs(0.4, 0.5, 1, audio_event=0.8, motion_events=1.0, motion_events_detail="2 прыжка"), weights)
    used = weights.weight_motion_intensity + weights.weight_scene_change + weights.weight_face_prominence + weights.weight_audio_event
    assert boosted.part("motion_events").points == pytest.approx(weights.weight_motion_events / used * 100)
    assert boosted.score > base.score
    # без бонуса оценка окна не зависит от того, проверяли ли его: вес бонуса не входит в знаменатель
    assert compute_viral_score(ScoreInputs(0.4, 0.5, 1), weights) == compute_viral_score(ScoreInputs(0.4, 0.5, 1, motion_events=0.0), weights)
    assert compute_score_breakdown(ScoreInputs(1, 1, 5, audio_event=1.0, motion_events=1.0), weights).score == 100   # потолок


def _window(start, presence, score=60):
    inputs = ScoreInputs(0.3, presence, 0)
    breakdown = compute_score_breakdown(inputs, UserSettings().viral_score)
    return WindowScore(start, start + 3.0, breakdown.score, 0.3, (), inputs, breakdown)


def test_runner_raises_only_windows_with_events_and_skips_windows_without_animals(tmp_path):
    video = _make_video(tmp_path / "jump.mp4", 6.0, lambda t: _jumping(t) if t < 3 else 170)   # прыжок в 1-м окне
    windows = [_window(0.0, presence=1.0), _window(3.0, presence=1.0), _window(6.0, presence=0.0)]
    runner = PipelineRunner(models_dir=tmp_path, output_dir=tmp_path)
    result = runner._add_motion_events(video, windows, _DetectorHandle(_BlobDetector()), [], UserSettings().viral_score)
    assert result[0].viral_score > windows[0].viral_score and result[0].inputs.motion_events >= 0.5
    assert "прыж" in result[0].inputs.motion_events_detail and result[0].breakdown.part("motion_events").points > 0
    assert result[1].viral_score == windows[1].viral_score and result[2] is windows[2]

    unavailable = runner._add_motion_events(video, windows, _DetectorHandle(None), [], UserSettings().viral_score)
    assert unavailable is windows                                    # без YOLO ничего не меняется
    no_weight = UserSettings().viral_score.__class__(weight_motion_events=0.0)
    assert runner._add_motion_events(video, windows, _DetectorHandle(_BlobDetector()), [], no_weight) is windows
