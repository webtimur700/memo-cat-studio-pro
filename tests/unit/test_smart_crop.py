from core.entities.detection import BoundingBox, Detection
from vision.smart_crop import SmartCropPlanner

SRC_W, SRC_H = 1920, 1080


def _det(cx, cy, w, h, t=0.0):
    return Detection(15, "cat", 0.9, BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), t)


def test_no_detection_crops_center():
    planner = SmartCropPlanner(SRC_W, SRC_H)
    window = planner.plan_frame(None)
    expected_w = SRC_H * (9 / 16)
    assert abs(window.width - expected_w) < 1.0
    assert abs((window.x1 + window.x2) / 2 - SRC_W / 2) < 1.0


def test_aspect_ratio_always_9_16():
    planner = SmartCropPlanner(SRC_W, SRC_H)
    window = planner.plan_frame(_det(960, 540, 400, 400))
    assert abs(window.width / window.height - 9 / 16) < 0.01


def test_small_subject_triggers_zoom():
    planner = SmartCropPlanner(SRC_W, SRC_H)
    small_cat = _det(960, 540, 80, 80)
    for _ in range(30):
        window = planner.plan_frame(small_cat)
    assert 1.5 < window.zoom_factor <= 2.2


def test_crop_clamped_within_frame_bounds():
    planner = SmartCropPlanner(SRC_W, SRC_H)
    edge_cat = _det(50, 50, 100, 100)
    for _ in range(30):
        window = planner.plan_frame(edge_cat)
    assert window.x1 >= 0 and window.y1 >= 0
    assert window.x2 <= SRC_W and window.y2 <= SRC_H


def test_smoothing_prevents_instant_jump():
    planner = SmartCropPlanner(SRC_W, SRC_H, smoothing_alpha=0.15)
    planner.plan_frame(_det(500, 540, 300, 300, t=0.0))
    window = planner.plan_frame(_det(1500, 540, 300, 300, t=0.1))
    center = (window.x1 + window.x2) / 2
    assert 500 < center < 1500
