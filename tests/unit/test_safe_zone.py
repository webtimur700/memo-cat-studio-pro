import numpy as np
from PIL import Image

from core.entities.detection import BoundingBox
from core.entities.settings import SafeZoneSettings, UserSettings
from effects.banner_text import draw_banner_text, split_runs
from effects.font_utils import find_emoji_font
from effects.safe_zone import SafeZone
from export.dynamic_crop import CropSample
from pipeline.pipeline_runner import PipelineRunner
from subtitles.ass_renderer import estimate_subtitle_band_height, render_ass
from core.entities.subtitle import SubtitleSegment, WordTiming
from vision.smart_crop import CropWindow

FRAME = (1080, 1920)


def test_zone_from_default_settings():
    zone = SafeZone.from_settings(FRAME, SafeZoneSettings())
    assert (zone.x1, zone.y1, zone.x2, zone.y2) == (60, 250, 930, 1470)
    assert zone.subtitle_margins(FRAME) == (60, 150, 450)


def test_nonsense_margins_do_not_squeeze_the_frame():
    zone = SafeZone.from_settings(FRAME, SafeZoneSettings(top_px=900, bottom_px=900))
    assert (zone.x1, zone.y1, zone.x2, zone.y2) == (0, 0, 1080, 1920)


def test_zone_scales_with_resolution():
    zone = SafeZone.from_settings((540, 960), SafeZoneSettings())
    assert (zone.x1, zone.y1, zone.y2) == (30, 125, 960 - 225)


def test_ass_uses_safe_zone_margins():
    seg = SubtitleSegment(words=[WordTiming("привет", 0.0, 0.5)])
    ass = render_ass([seg], margin_l=60, margin_r=150, margin_v=450)
    style_line = next(l for l in ass.splitlines() if l.startswith("Style:"))
    assert style_line.endswith(",2,60,150,450,1")


def test_split_runs_separates_emoji_and_drops_selectors():
    assert split_runs("💼 100zarplat.ru") == [("💼", True), (" 100zarplat.ru", False)]
    assert split_runs("❤️ ok") == [("❤", True), (" ok", False)]


def test_banner_text_draws_emoji_pixels():
    if find_emoji_font() is None:
        import pytest

        pytest.skip("нет рабочего эмодзи-шрифта")
    with_emoji = Image.new("RGBA", (900, 220), (0, 0, 0, 0))
    plain = Image.new("RGBA", (900, 220), (0, 0, 0, 0))
    draw_banner_text(with_emoji, (0, 0, 900, 220), ["💼 100zarplat.ru"], 1.0)
    draw_banner_text(plain, (0, 0, 900, 220), ["100zarplat.ru"], 1.0)
    arr = np.asarray(with_emoji).astype(int)
    # цветные (не белые) пиксели с альфой — это эмодзи
    colored = (arr[..., 3] > 200) & ((arr[..., 0] != arr[..., 2]) | (arr[..., 1] != arr[..., 2]))
    assert colored.any()
    plain_arr = np.asarray(plain).astype(int)
    assert not ((plain_arr[..., 3] > 200) & (plain_arr[..., 0] != plain_arr[..., 2])).any()


def test_banner_text_fits_narrow_banner():
    layer = Image.new("RGBA", (1080, 220), (0, 0, 0, 0))
    lines = ["💼 100zarplat.ru", "Подбор вакансий от проверенных работодателей", "💰 Ежедневные и еженедельные выплаты"]
    draw_banner_text(layer, (100, 0, 800, 220), lines, 1.0, start_font_size=40)   # 700 px шириной
    bbox = layer.getchannel("A").getbbox()
    assert bbox[0] >= 100 and bbox[2] <= 800   # текст не вылез за плашку


def _samples():
    win = CropWindow(0, 0, 540, 960, 1.0, 0.0)
    return [CropSample(t, CropWindow(win.x1, win.y1, win.x2, win.y2, 1.0, t)) for t in (0.0, 5.0, 10.0)]


def test_banner_stays_inside_safe_zone(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    zone = SafeZone.from_settings(FRAME, settings.safe_zone)
    x1, y1, x2, y2 = runner._build_banner_rect(settings, _samples(), [None] * 3, None, 4.0, 5.0, zone=zone)
    assert zone.x1 <= x1 and x2 <= zone.x2 and zone.y1 <= y1 and y2 <= zone.y2
    assert y2 == zone.y2                              # прижата к низу безопасной зоны, а не к краю кадра (было y=1660)


def test_banner_rises_above_subtitles_when_they_overlap_in_time(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    zone = SafeZone.from_settings(FRAME, settings.safe_zone)
    without = runner._build_banner_rect(settings, _samples(), [None] * 3, None, 4.0, 5.0, zone=zone)
    with_subs = runner._build_banner_rect(
        settings, _samples(), [None] * 3, None, 4.0, 5.0, zone=zone, subtitles_during_banner=True
    )
    band = estimate_subtitle_band_height(settings.subtitles.style_preset)
    subtitle_top = zone.y2 - band
    assert with_subs[3] <= subtitle_top               # плашка целиком выше полосы субтитров
    assert with_subs[1] < without[1]


def test_animal_head_and_subtitles_and_logo_all_avoided(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    zone = SafeZone.from_settings(FRAME, settings.safe_zone)
    head = BoundingBox(50, 500, 450, 650)             # -> y 1000..1300 в выходном кадре, на месте поднятой плашки
    rect = runner._build_banner_rect(
        settings, _samples(), [head] * 3, None, 4.0, 5.0, zone=zone, subtitles_during_banner=True
    )
    assert zone.y1 <= rect[1] and rect[3] <= zone.y2
    assert rect[3] <= 1000 or rect[1] >= 1300         # не пересекает голову
