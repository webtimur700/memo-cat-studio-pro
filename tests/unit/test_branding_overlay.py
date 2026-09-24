from pathlib import Path

from PIL import Image

from effects.branding_overlay import (
    BrandingOverlay,
    create_default_logo,
    logo_xy,
    resolve_logo_path,
    subscribe_xy,
)


def test_user_logo_takes_priority_over_default(tmp_path):
    logo_dir = tmp_path / "logo"
    logo_dir.mkdir()
    Image.new("RGBA", (10, 10), (1, 2, 3, 255)).save(logo_dir / "logo.png")
    create_default_logo(logo_dir / "default_logo.png")
    assert resolve_logo_path(tmp_path) == logo_dir / "logo.png"


def test_default_logo_is_generated_when_missing(tmp_path):
    path = resolve_logo_path(tmp_path)
    assert path == tmp_path / "logo" / "default_logo.png"
    with Image.open(path) as image:
        assert image.mode == "RGBA" and image.getbbox() is not None


def test_logo_position_corners():
    frame, sprite = (1080, 1920), (200, 80)
    assert logo_xy("top_right", frame, sprite)[0] > frame[0] // 2
    assert logo_xy("top_left", frame, sprite)[0] < frame[0] // 2
    assert logo_xy("bottom_left", frame, sprite)[1] > frame[1] // 2
    # кнопка — на противоположной от логотипа стороне
    assert subscribe_xy("top_right", frame, (300, 80))[0] < frame[0] // 2
    assert subscribe_xy("top_left", frame, (300, 80))[0] > frame[0] // 2


def test_animation_timeline(tmp_path):
    overlay = BrandingOverlay.build((1080, 1920), resolve_logo_path(tmp_path), "top_right", True)

    def alpha_bbox(t):
        canvas = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
        overlay.draw_on(canvas, t)
        return canvas.getchannel("A").getbbox()

    assert alpha_bbox(0.0) is None                      # до старта пусто
    assert alpha_bbox(3.0) is not None                  # логотип + кнопка
    left_half = Image.new("RGBA", (1080, 1920))
    overlay.draw_on(left_half, 3.0)
    assert left_half.crop((0, 0, 540, 1920)).getchannel("A").getbbox() is not None  # кнопка слева
    assert alpha_bbox(20.0) is not None                 # логотип остаётся постоянно
    canvas = Image.new("RGBA", (1080, 1920))
    overlay.draw_on(canvas, 20.0)
    assert canvas.crop((0, 0, 540, 1920)).getchannel("A").getbbox() is None  # кнопка ушла


def test_disabled_subscribe_and_missing_logo():
    overlay = BrandingOverlay.build((1080, 1920), None, "top_right", False)
    assert overlay.is_empty
