from core.entities.detection import BoundingBox, Detection
from vision.tracker import ObjectTracker


def _det(cls_id, cls_name, x, y, w=50, h=50, t=0.0):
    return Detection(cls_id, cls_name, 0.9, BoundingBox(x, y, x + w, y + h), t)


def test_track_persists_through_smooth_motion():
    tracker = ObjectTracker(max_missed_frames=3)
    for i in range(5):
        tracker.update([_det(15, "cat", 100 + i * 10, 100, t=i)])
    assert len(tracker.tracks) == 1


def test_track_survives_short_detector_gap():
    tracker = ObjectTracker(max_missed_frames=3)
    tracker.update([_det(15, "cat", 100, 100, t=0)])
    track_id_before = tracker.tracks[0].track_id

    for _ in range(3):
        tracker.update([])
    assert len(tracker.tracks) == 1

    tracker.update([_det(15, "cat", 115, 100, t=4)])
    assert tracker.tracks[0].track_id == track_id_before


def test_different_classes_get_separate_tracks():
    tracker = ObjectTracker()
    tracker.update([_det(15, "cat", 100, 100, t=0), _det(16, "dog", 400, 400, t=0)])
    assert len(tracker.tracks) == 2


def test_track_removed_after_long_absence():
    tracker = ObjectTracker(max_missed_frames=3)
    tracker.update([_det(15, "cat", 100, 100, t=0)])
    for i in range(5):
        tracker.update([_det(16, "dog", 400 + i, 400, t=1 + i)])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].last_detection.class_name == "dog"
