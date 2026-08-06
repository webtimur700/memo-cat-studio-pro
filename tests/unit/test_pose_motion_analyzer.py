from core.entities.detection import BoundingBox, Detection
from vision.pose_motion_analyzer import MotionEventType, PoseMotionAnalyzer


def _det(cx, cy, t, w=70, h=70):
    return Detection(15, "cat", 0.9, BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), t)


def test_jump_is_detected():
    analyzer = PoseMotionAnalyzer()
    track = [_det(500, 300, 0.0), _det(500, 250, 0.2), _det(500, 300, 0.4)]
    result = analyzer.analyze(track)
    assert any(e.event_type == MotionEventType.JUMP for e in result.events)


def test_fall_is_detected():
    analyzer = PoseMotionAnalyzer()
    track = [_det(500, 300, 0.0), _det(500, 400, 0.2), _det(500, 401, 0.4)]
    result = analyzer.analyze(track)
    assert any(e.event_type == MotionEventType.FALL for e in result.events)


def test_calm_walk_produces_no_events():
    analyzer = PoseMotionAnalyzer()
    track = [_det(500 + i * 3, 300, i * 0.2) for i in range(6)]
    result = analyzer.analyze(track)
    assert result.events == []
    assert result.motion_intensity < 0.3


def test_rapid_horizontal_movement_detected():
    analyzer = PoseMotionAnalyzer()
    track = [_det(500 + i * 100, 300, i * 0.2) for i in range(5)]
    result = analyzer.analyze(track)
    assert any(e.event_type == MotionEventType.RAPID_MOVEMENT for e in result.events)
