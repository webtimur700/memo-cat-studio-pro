"""Цветные эмодзи как картинки: общий рендер для обложки и рекламной плашки.

Эмодзи рисуются отдельным цветным шрифтом (не тем же вызовом, что и текст).
Шрифт выбирает effects/font_utils.find_emoji_font() — только тот, что Pillow
реально рисует (на Fedora это Twemoji; системный Noto COLRv1 Pillow не умеет).
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from effects.font_utils import find_emoji_font

_FONT_CACHE: dict[str, ImageFont.FreeTypeFont] = {}


def get_native_emoji_font() -> ImageFont.FreeTypeFont | None:
    """NotoColorEmoji — bitmap-strike шрифт: он НЕ поддерживает произвольный
    размер через truetype(..., size) (реальная ошибка "invalid pixel size",
    пойманная тестом на этом шаге) — доступен только фиксированный набор
    "strikes". Пробуем нативные размеры по убыванию, кэшируем первый рабочий.
    Итоговый размер под style.emoji_size достигается последующим resize()
    уже отрендеренного тайла, а не запросом размера у самого шрифта.
    """
    if "font" in _FONT_CACHE:
        return _FONT_CACHE["font"]  # type: ignore[return-value]

    # find_emoji_font() принимает шрифт только после пробного рендера: системный
    # Noto Color Emoji на Fedora — COLRv1, Pillow его открывает, но рисует пустоту.
    found = find_emoji_font()
    if found is None:
        return None
    path, size = found
    try:
        font = ImageFont.truetype(path, size)
    except OSError:
        return None
    _FONT_CACHE["font"] = font
    return font


def render_emoji_tile(char: str, target_size: int) -> Image.Image | None:
    font = get_native_emoji_font()
    if font is None:
        return None

    probe = Image.new("RGBA", (font.size * 2, font.size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), char, font=font, embedded_color=True)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None  # глиф отсутствует даже в эмодзи-шрифте

    draw.text((0, 0), char, font=font, embedded_color=True)
    tile = probe.crop(bbox)

    scale = target_size / max(tile.width, tile.height)
    new_size = (max(1, int(tile.width * scale)), max(1, int(tile.height * scale)))
    return tile.resize(new_size, Image.Resampling.LANCZOS)




_TILE_CACHE: dict[tuple[str, int], Image.Image | None] = {}


def cached_emoji_tile(char: str, target_size: int) -> Image.Image | None:
    """render_emoji_tile с кэшем: плашка рисуется на сотнях кадров подряд."""
    key = (char, target_size)
    if key not in _TILE_CACHE:
        _TILE_CACHE[key] = render_emoji_tile(char, target_size)
    return _TILE_CACHE[key]
