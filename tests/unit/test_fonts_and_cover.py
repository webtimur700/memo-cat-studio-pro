import numpy as np
import pytest

from effects.cover_generator import generate_cover
from effects.font_utils import (
    DEFAULT_FALLBACK_FONT,
    find_emoji_font,
    find_font_file,
    font_supports_text,
    preferred_title_fonts,
)


def test_fallback_font_exists_and_covers_cyrillic():
    assert font_supports_text(DEFAULT_FALLBACK_FONT, "Привет, мир! Hello 123")


def test_find_font_file_rejects_substituted_family():
    # fc-match подставил бы другой шрифт; это не должно выдаваться за запрошенное семейство
    assert find_font_file("Definitely Not A Real Font Family 42") is None


def test_preferred_title_fonts_exist():
    from pathlib import Path

    assert all(Path(p).is_file() for p in preferred_title_fonts())


def test_found_emoji_font_really_renders():
    found = find_emoji_font()
    if found is None:
        pytest.skip("в системе нет эмодзи-шрифта, который умеет рисовать Pillow")
    from PIL import Image, ImageDraw, ImageFont

    path, size = found
    image = Image.new("RGBA", (size * 3, size * 3), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((0, 0), "🔥", font=ImageFont.truetype(path, size), embedded_color=True)
    assert image.getbbox() is not None


def test_generate_cover_draws_title_and_emoji_row():
    frame = np.full((1920, 1080, 3), 40, dtype=np.uint8)
    plain = generate_cover(frame, "Кот НЕ ОЖИДАЛ такого", emoji=None).convert("RGB")
    with_emoji = generate_cover(frame, "Кот НЕ ОЖИДАЛ такого", emoji="🔥").convert("RGB")
    assert plain.size == (1080, 1920)
    # текст нарисован: в нижней части есть белые пиксели заголовка
    assert (np.asarray(plain)[1300:1800] == 255).all(axis=2).any()
    if find_emoji_font() is not None:
        # эмодзи-ряд добавляет цветные (не серые) пиксели
        arr = np.asarray(with_emoji).astype(int)
        assert (np.abs(arr[..., 0] - arr[..., 2]) > 60).any()
