from pathlib import Path

from core.entities.detection import BoundingBox, Detection
from core.entities.settings import UserSettings
from effects.branding_overlay import BrandingOverlay, resolve_logo_path
from effects.collision_detector import BannerPosition, resolve_banner_position_over_time
from export.dynamic_crop import CropSample
from pipeline.pipeline_runner import PipelineRunner, _motion_direction_x
from vision.smart_crop import CropWindow

W, H, BW, BH = 1080, 1920, 918, 220


def test_no_obstacles_keeps_preferred_position():
    result = resolve_banner_position_over_time(BannerPosition.BOTTOM_CENTER, [], W, H, BW, BH)
    assert result.final_position == BannerPosition.BOTTOM_CENTER and not result.was_repositioned


def test_animal_at_bottom_moves_banner_to_top():
    animal_low = BoundingBox(200, 1600, 900, 1850)
    result = resolve_banner_position_over_time(BannerPosition.BOTTOM_CENTER, [animal_low], W, H, BW, BH)
    assert result.final_position == BannerPosition.TOP_CENTER and result.was_repositioned


def test_any_moment_of_collision_counts_not_just_first():
    early_free = BoundingBox(0, 900, 100, 1000)      # не пересекает нижнюю плашку
    late_low = BoundingBox(200, 1700, 900, 1900)     # животное спустилось вниз позже
    result = resolve_banner_position_over_time(BannerPosition.BOTTOM_CENTER, [early_free, late_low], W, H, BW, BH)
    assert result.final_position != BannerPosition.BOTTOM_CENTER


def test_everything_blocked_picks_least_bad_not_blindly_preferred():
    # нижняя часть занята целиком, верхняя — только узкой полоской
    obstacles = [BoundingBox(0, 1000, W, H), BoundingBox(0, 40, 60, 260)]
    result = resolve_banner_position_over_time(BannerPosition.BOTTOM_CENTER, obstacles, W, H, BW, BH)
    assert result.final_position in (BannerPosition.TOP_CENTER, BannerPosition.TOP_LEFT, BannerPosition.TOP_RIGHT)


def test_motion_direction_from_track():
    def det(x1, x2):
        return Detection(15, "cat", 0.9, BoundingBox(x1, 0, x2, 100), 0.0)

    assert _motion_direction_x([det(0, 100)]) == 0.0
    assert _motion_direction_x([det(0, 100), det(50, 150)]) == 0.5
    assert _motion_direction_x([det(0, 100), det(-500, -400)]) == -1.0


def _samples(window: CropWindow):
    return [CropSample(t, CropWindow(window.x1, window.y1, window.x2, window.y2, 1.0, t)) for t in (0.0, 5.0, 10.0)]


def test_head_region_mapped_from_source_to_output_coordinates(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    # окно кропа 540x960 в исходном кадре, начиная с (100, 200): масштаб x2 до 1080x1920
    samples = _samples(CropWindow(100, 200, 640, 1160, 1.0, 0.0))
    head = BoundingBox(200, 300, 300, 400)   # внутри окна
    obstacles = runner._banner_obstacles(settings, samples, [head, head, head], None, 4.0, 5.0)
    assert len(obstacles) == 1                # только сэмпл t=5.0 попадает в окно показа 4..9
    ob = obstacles[0]
    assert (ob.x1, ob.y1, ob.x2, ob.y2) == (200.0, 200.0, 400.0, 400.0)


def test_head_outside_crop_window_is_ignored(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    samples = _samples(CropWindow(100, 200, 640, 1160, 1.0, 0.0))
    far_head = BoundingBox(1500, 300, 1600, 400)
    assert runner._banner_obstacles(UserSettings(), samples, [far_head] * 3, None, 4.0, 5.0) == []


def test_banner_avoids_animal_low_in_frame(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    samples = _samples(CropWindow(0, 0, 540, 960, 1.0, 0.0))
    low_head = BoundingBox(50, 800, 450, 950)   # -> внизу выходного кадра
    rect = runner._build_banner_rect(settings, samples, [low_head] * 3, None, 4.0, 5.0)
    assert rect[1] < H // 2                     # плашка ушла наверх


def test_collision_avoidance_can_be_disabled(tmp_path):
    from dataclasses import replace

    runner = PipelineRunner(output_dir=tmp_path)
    base = UserSettings()
    settings = replace(base, branding=replace(base.branding, collision_avoidance=False))
    samples = _samples(CropWindow(0, 0, 540, 960, 1.0, 0.0))
    rect = runner._build_banner_rect(settings, samples, [BoundingBox(50, 800, 450, 950)] * 3, None, 4.0, 5.0)
    assert rect[1] > H // 2


def test_logo_and_subscribe_are_obstacles(tmp_path):
    branding = BrandingOverlay.build((W, H), resolve_logo_path(tmp_path), "top_right", True)
    runner = PipelineRunner(output_dir=tmp_path)
    obstacles = runner._banner_obstacles(UserSettings(), [], [], branding, 4.0, 5.0)
    assert len(obstacles) == 2                  # логотип + Subscribe (виден 1..7 с)
    assert len(branding.obstacle_rects(8.0, 9.0)) == 2 or len(branding.obstacle_rects(8.0, 9.0)) == 1
    assert len(branding.obstacle_rects(7.5, 9.0)) == 1  # Subscribe уже ушёл
