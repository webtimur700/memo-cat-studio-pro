"""Генерация обложки Shorts (Функция 11): лучший кадр (уже выбран
video/frame_extractor.best_frame_for_cover) + крупный текст + эмодзи +
эффект свечения вокруг текста.

ВАЖНО (реальная проблема, пойманная тестом на этом шаге и исправленная):
"модные" шрифты вроде Poppins часто не содержат кириллицу. Шрифт для текста
выбирается через effects/font_utils.resolve_font_path() — реальная проверка
cmap-таблицы, а не просто "нравится ли шрифт эстетически". Эмодзи — ВСЕГДА
отдельный цветной шрифт (NotoColorEmoji), их нельзя рисовать тем же вызовом,
что и обычный текст.

Layout: крупный эмодзи-ряд сверху текстового блока, затем перенесённый по
словам заголовок с обводкой и glow снизу — устойчивый паттерн для обложек
Shorts, не требующий сложного посимвольного шейпинга смешанных шрифтов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from effects.emoji_render import render_emoji_tile
from effects.font_utils import (
    DEFAULT_FALLBACK_FONT,
    is_emoji,
    preferred_title_fonts,
    resolve_font_path,
)

PREFERRED_TITLE_FONTS = preferred_title_fonts()


@dataclass(frozen=True, slots=True)
class CoverStyle:
    font_size: int = 92
    emoji_size: int = 120
    text_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    outline_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    outline_width: int = 6
    glow_color: tuple[int, int, int, int] = (124, 92, 255, 160)
    glow_blur_radius: int = 22
    max_text_width_ratio: float = 0.85
    text_vertical_position_ratio: float = 0.80
    darken_background_alpha: int = 80


def _extract_emoji_and_text(raw_text: str) -> tuple[list[str], str]:
    """Разделяет входную строку на список эмодзи-символов и "чистый" текст
    (без эмодзи, с нормализованными пробелами) — раздельный рендер, см. docstring модуля.
    """
    emojis = [ch for ch in raw_text if is_emoji(ch)]
    text_only = "".join(ch for ch in raw_text if not is_emoji(ch))
    text_only = re.sub(r"\s+", " ", text_only).strip()
    return emojis, text_only


def _wrap_text_to_width(
    text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw, max_width: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def generate_cover(
    frame_bgr: np.ndarray,
    title_text: str,
    emoji: str | None = "🔥",
    style: CoverStyle | None = None,
) -> Image.Image:
    style = style or CoverStyle()

    frame_rgb = frame_bgr[:, :, ::-1]
    base = Image.fromarray(frame_rgb, mode="RGB").convert("RGBA")
    width, height = base.size

    extra_emojis, clean_title = _extract_emoji_and_text(f"{emoji or ''} {title_text}")

    title_font_path = resolve_font_path(clean_title, PREFERRED_TITLE_FONTS, DEFAULT_FALLBACK_FONT)
    title_font = ImageFont.truetype(title_font_path, style.font_size)

    draw_probe = ImageDraw.Draw(base)
    max_text_width = int(width * style.max_text_width_ratio)
    lines = _wrap_text_to_width(clean_title, title_font, draw_probe, max_text_width)

    line_heights = [
        draw_probe.textbbox((0, 0), line, font=title_font)[3]
        - draw_probe.textbbox((0, 0), line, font=title_font)[1]
        for line in lines
    ]
    line_spacing = int(style.font_size * 0.25)
    emoji_row_height = style.emoji_size + 16 if extra_emojis else 0
    total_text_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1) + emoji_row_height

    block_top = int(height * style.text_vertical_position_ratio) - total_text_height // 2
    block_top = max(20, min(block_top, height - total_text_height - 20))

    # 1. Затемнение области под текстом/эмодзи — читаемость на любом фоне.
    darken_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(darken_layer).rectangle(
        (0, block_top - 24, width, block_top + total_text_height + 24),
        fill=(0, 0, 0, style.darken_background_alpha),
    )
    base = Image.alpha_composite(base, darken_layer)

    y_cursor = block_top

    # 2. Ряд эмодзи (цветной шрифт, рендерится ОТДЕЛЬНО от текста).
    if extra_emojis:
        try:
            emoji_tiles = [render_emoji_tile(ch, style.emoji_size) for ch in extra_emojis]
            emoji_tiles = [t for t in emoji_tiles if t is not None]

            if emoji_tiles:
                total_emoji_width = sum(t.width for t in emoji_tiles) + 12 * (len(emoji_tiles) - 1)
                x_cursor = (width - total_emoji_width) // 2
                for tile in emoji_tiles:
                    base.alpha_composite(tile, (x_cursor, y_cursor))
                    x_cursor += tile.width + 12
        except Exception:
            # Если системный эмодзи-шрифт недоступен на целевой машине —
            # обложка всё равно рендерится, просто без эмодзи-ряда, а не падает.
            pass
        y_cursor += emoji_row_height

    # 3. Glow-слой под текстом.
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    gy = y_cursor
    for line, line_height in zip(lines, line_heights):
        line_width = glow_draw.textbbox((0, 0), line, font=title_font)[2]
        x = (width - line_width) // 2
        glow_draw.text((x, gy), line, font=title_font, fill=style.glow_color)
        gy += line_height + line_spacing
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(style.glow_blur_radius))
    base = Image.alpha_composite(base, glow_layer)

    # 4. Основной текст с обводкой.
    text_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    ty = y_cursor
    for line, line_height in zip(lines, line_heights):
        line_width = text_draw.textbbox((0, 0), line, font=title_font)[2]
        x = (width - line_width) // 2
        text_draw.text(
            (x, ty),
            line,
            font=title_font,
            fill=style.text_color,
            stroke_width=style.outline_width,
            stroke_fill=style.outline_color,
        )
        ty += line_height + line_spacing
    base = Image.alpha_composite(base, text_layer)

    return base
