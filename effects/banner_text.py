"""Текст рекламной плашки: строки с цветными эмодзи прямо в тексте (💼 100zarplat.ru).

Обычный текст рисуется шрифтом с кириллицей, эмодзи — тайлами из
effects/emoji_render (тот же Twemoji-рендер, что и у обложки). Кегль
подбирается так, чтобы самая длинная строка помещалась в плашку.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from effects.emoji_render import cached_emoji_tile
from effects.font_utils import DEFAULT_FALLBACK_FONT, is_emoji, resolve_font_path

_IGNORED = {"️", "︎", "‍"}   # variation selectors и ZWJ — отдельного глифа не имеют
EMOJI_SCALE = 1.15
LINE_HEIGHT_RATIO = 1.3
MIN_FONT_SIZE = 18
WIDTH_FILL = 0.92
HEIGHT_FILL = 0.90


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def split_runs(line: str) -> list[tuple[str, bool]]:
    """[(текст, это_эмодзи)] — подряд идущий обычный текст склеивается в один run."""
    runs: list[tuple[str, bool]] = []
    for ch in line:
        if ch in _IGNORED:
            continue
        emoji = is_emoji(ch)
        if runs and not emoji and not runs[-1][1]:
            runs[-1] = (runs[-1][0] + ch, False)
        else:
            runs.append((ch, emoji))
    return runs


def _line_width(line: str, font: ImageFont.FreeTypeFont, font_size: int, draw: ImageDraw.ImageDraw) -> float:
    total = 0.0
    for text, emoji in split_runs(line):
        if emoji:
            tile = cached_emoji_tile(text, int(font_size * EMOJI_SCALE))
            total += tile.width if tile is not None else 0
        else:
            total += draw.textlength(text, font=font)
    return total


def fit_font_size(lines: list[str], font_path: str, max_width: float, max_height: float, start_size: int) -> int:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    size = start_size
    while size > MIN_FONT_SIZE:
        font = _font(font_path, size)
        widest = max((_line_width(line, font, size, probe) for line in lines), default=0.0)
        if widest <= max_width and size * LINE_HEIGHT_RATIO * len(lines) <= max_height:
            return size
        size -= 2
    return MIN_FONT_SIZE


def draw_banner_text(
    layer: Image.Image, rect: tuple[int, int, int, int], lines: list[str], opacity: float, start_font_size: int = 40
) -> None:
    """Рисует строки по центру rect на прозрачном слое (изменяется на месте)."""
    if not lines:
        return
    x1, y1, x2, y2 = rect
    plain_text = " ".join(t for line in lines for t, emoji in split_runs(line) if not emoji)
    font_path = resolve_font_path(plain_text, [], DEFAULT_FALLBACK_FONT)
    font_size = fit_font_size(lines, font_path, (x2 - x1) * WIDTH_FILL, (y2 - y1) * HEIGHT_FILL, start_font_size)
    font = _font(font_path, font_size)

    draw = ImageDraw.Draw(layer)
    line_height = font_size * LINE_HEIGHT_RATIO
    y_center = y1 + ((y2 - y1) - line_height * len(lines)) / 2 + line_height / 2
    alpha = int(255 * opacity)

    for line in lines:
        x = x1 + ((x2 - x1) - _line_width(line, font, font_size, draw)) / 2
        for text, emoji in split_runs(line):
            if emoji:
                tile = cached_emoji_tile(text, int(font_size * EMOJI_SCALE))
                if tile is None:
                    continue
                if opacity < 0.995:
                    tile = tile.copy()
                    tile.putalpha(tile.getchannel("A").point(lambda v: int(v * opacity)))
                layer.alpha_composite(tile, (int(x), int(y_center - tile.height / 2)))
                x += tile.width
            else:
                draw.text((x, y_center), text, font=font, fill=(255, 255, 255, alpha), anchor="lm")
                x += draw.textlength(text, font=font)
        y_center += line_height
