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


def test_animal_takes_priority_over_longer_person_track():
    tracker = ObjectTracker()
    for i in range(10):
        dets = [_det(0, "person", 10 + i, 10, w=300, h=600, t=i)]
        if i >= 7:  # кот появился позже и виден короче
            dets.append(_det(15, "cat", 500, 500, t=i))
        tracker.update(dets)
    primary = tracker.primary_track()
    assert primary.last_detection.class_name == "cat"


def test_recently_lost_animal_still_beats_person():
    tracker = ObjectTracker(max_missed_frames=8)
    for i in range(3):
        tracker.update([_det(16, "dog", 100, 100, t=i), _det(0, "person", 600, 100, t=i)])
    for i in range(3, 6):  # собака на пару кадров пропала, человек остался
        tracker.update([_det(0, "person", 600, 100, t=i)])
    assert tracker.primary_track().last_detection.class_name == "dog"


def test_falls_back_to_person_when_no_animals():
    tracker = ObjectTracker()
    tracker.update([_det(0, "person", 100, 100)])
    assert tracker.primary_track().last_detection.class_name == "person"


def test_all_coco_animals_are_known_classes():
    from core.entities.detection import is_animal_class
    from vision.yolo_detector import COCO_CLASS_NAMES

    for class_id in (14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
        assert is_animal_class(class_id) and class_id in COCO_CLASS_NAMES
    assert not is_animal_class(0) and COCO_CLASS_NAMES[0] == "person"


def test_animal_class_flip_keeps_single_track():
    tracker = ObjectTracker()
    for i, (cid, name) in enumerate([(15, "cat"), (14, "bird"), (16, "dog"), (15, "cat")]):
        tracker.update([_det(cid, name, 100 + i * 5, 100, t=i)])
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].detections) == 4


def test_person_and_animal_are_not_merged():
    tracker = ObjectTracker()
    tracker.update([_det(0, "person", 100, 100)])
    tracker.update([_det(15, "cat", 100, 100)])
    assert len(tracker.tracks) == 2
