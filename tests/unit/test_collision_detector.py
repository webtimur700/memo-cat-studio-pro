from core.entities.detection import BoundingBox
from effects.collision_detector import BannerPosition, compute_banner_rect, resolve_banner_position

FRAME_W, FRAME_H = 1080, 1920
BANNER_W, BANNER_H = 880, 260


def test_no_reposition_when_no_overlap():
    animal_top = BoundingBox(300, 100, 780, 700)
    result = resolve_banner_position(BannerPosition.BOTTOM_CENTER, animal_top, FRAME_W, FRAME_H, BANNER_W, BANNER_H)
    assert result.final_position == BannerPosition.BOTTOM_CENTER
    assert result.was_repositioned is False


def test_reposition_on_overlap_and_new_position_is_free():
    animal_bottom = BoundingBox(200, 1500, 900, 1880)
    result = resolve_banner_position(BannerPosition.BOTTOM_CENTER, animal_bottom, FRAME_W, FRAME_H, BANNER_W, BANNER_H)
    assert result.was_repositioned is True
    assert result.final_position != BannerPosition.BOTTOM_CENTER

    rect = result.final_rect

    def overlap_ratio(a, b):
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / a.area

    assert overlap_ratio(rect, animal_bottom) < 0.02


def test_no_animal_detection_keeps_preferred_position():
    result = resolve_banner_position(BannerPosition.BOTTOM_CENTER, None, FRAME_W, FRAME_H, BANNER_W, BANNER_H)
    assert result.was_repositioned is False
    assert result.final_position == BannerPosition.BOTTOM_CENTER


def test_compute_banner_rect_top_left():
    rect = compute_banner_rect(BannerPosition.TOP_LEFT, FRAME_W, FRAME_H, BANNER_W, BANNER_H, margin=40)
    assert rect.x1 == 40 and rect.y1 == 40
    assert rect.width == BANNER_W and rect.height == BANNER_H
