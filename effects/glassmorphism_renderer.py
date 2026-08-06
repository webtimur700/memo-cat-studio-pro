"""Рендер glassmorphism-плашки в кадр видео (Функция 12: анимированная
рекламная плашка — "дорого", blur, glow, rounded corners, drop shadow).

Тот же визуальный язык, что и `ui/widgets/glass_panel.py` (Шаг 4), но здесь —
реальный растровый рендер через Pillow для композитинга поверх кадра видео,
а не Qt-виджет. Ключевая техника: полупрозрачная заливка + Gaussian Blur
самого фона под плашкой (настоящий "стеклянный" размытый фон, а не только
полупрозрачный цвет) + мягкая тень + светлая окантовка.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True, slots=True)
class GlassPanelStyle:
    corner_radius: int = 28
    fill_rgba: tuple[int, int, int, int] = (255, 255, 255, 30)
    border_rgba: tuple[int, int, int, int] = (255, 255, 255, 70)
    border_width: int = 2
    background_blur_radius: int = 18   # размытие ФОНА под плашкой ("настоящее стекло")
    shadow_blur_radius: int = 24
    shadow_rgba: tuple[int, int, int, int] = (0, 0, 0, 140)
    shadow_offset: tuple[int, int] = (0, 8)
    glow_rgba: tuple[int, int, int, int] = (124, 92, 255, 90)   # фирменный фиолетовый Memo Cat
    glow_radius: int = 20


def render_glass_panel_on_frame(
    frame: Image.Image,
    rect: tuple[int, int, int, int],  # (x1, y1, x2, y2) в пикселях кадра
    style: GlassPanelStyle | None = None,
) -> Image.Image:
    """Возвращает НОВОЕ изображение кадра с наложенной glassmorphism-плашкой.
    frame должен быть в режиме RGBA (конвертируется автоматически, если нет).
    """
    style = style or GlassPanelStyle()
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1

    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    result = frame.copy()

    # 1. Настоящее размытие фона под плашкой — вырезаем область, блюрим,
    #    вставляем обратно (это и есть "стекло", а не просто прозрачный цвет).
    region = result.crop((x1, y1, x2, y2))
    blurred_region = region.filter(ImageFilter.GaussianBlur(style.background_blur_radius))
    result.paste(blurred_region, (x1, y1))

    # 2. Мягкая тень позади плашки — рисуется на отдельном слое размером
    #    с кадр, чтобы блюр тени не обрезался границами плашки.
    shadow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    sx, sy = style.shadow_offset
    shadow_draw.rounded_rectangle(
        (x1 + sx, y1 + sy, x2 + sx, y2 + sy),
        radius=style.corner_radius,
        fill=style.shadow_rgba,
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(style.shadow_blur_radius))
    result = Image.alpha_composite(result, shadow_layer)

    # 3. Glow (фирменный цвет) — светящаяся окантовка чуть шире самой плашки.
    glow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.rounded_rectangle(
        (x1, y1, x2, y2), radius=style.corner_radius, outline=style.glow_rgba, width=6
    )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(style.glow_radius))
    result = Image.alpha_composite(result, glow_layer)

    # 4. Полупрозрачная заливка поверх размытого фона + окантовка — контур плашки.
    panel_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel_layer)
    panel_draw.rounded_rectangle((x1, y1, x2, y2), radius=style.corner_radius, fill=style.fill_rgba)
    panel_draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=style.corner_radius,
        outline=style.border_rgba,
        width=style.border_width,
    )
    result = Image.alpha_composite(result, panel_layer)

    return result
