"""Анимация рекламной плашки (Функция 12): появляется на 4-й секунде,
держится 5 секунд, состоит из трёх фаз:

  1. ENTER (первые ENTER_DURATION_SEC): Slide Left + Fade In + Scale 0.8->1 + Glow Pulse
  2. HOLD (середина): Floating — лёгкое покачивание вверх-вниз
  3. EXIT (последние EXIT_DURATION_SEC): Fade Out + Slide Right

compute_banner_animation_state() — чистая функция времени, полностью
детерминированная и тестируемая без рендера кадров: на вход время с начала
показа плашки, на выход — параметры трансформации (opacity, scale, x_offset,
y_offset, glow_intensity), которые дальше применяет
animation_render.render_banner_frame() (PIL) поверх кадра видео.
"""

from __future__ import annotations

from dataclasses import dataclass

from animation.easing import clamp01, ease_out_back, ease_out_cubic, sine_wave

ENTER_DURATION_SEC = 0.45
EXIT_DURATION_SEC = 0.35
SLIDE_DISTANCE_PX = 120
FLOAT_AMPLITUDE_PX = 6
FLOAT_FREQUENCY_HZ = 0.5
GLOW_PULSE_FREQUENCY_HZ = 1.2


@dataclass(frozen=True, slots=True)
class BannerAnimationState:
    visible: bool
    opacity: float          # 0..1
    scale: float             # относительно базового размера плашки
    x_offset_px: float       # смещение по X относительно целевой позиции (slide)
    y_offset_px: float       # смещение по Y (floating)
    glow_intensity: float    # 0..1, множитель альфы glow-слоя


def compute_banner_animation_state(
    elapsed_sec: float,
    total_duration_sec: float,
    enter_duration_sec: float = ENTER_DURATION_SEC,
    exit_duration_sec: float = EXIT_DURATION_SEC,
) -> BannerAnimationState:
    """elapsed_sec — время С МОМЕНТА появления плашки (0 = самый первый кадр
    показа), НЕ время с начала видео. Вызывающая сторона вычитает
    appear_at_sec из timestamp кадра перед вызовом.
    """
    if elapsed_sec < 0 or elapsed_sec > total_duration_sec:
        return BannerAnimationState(False, 0.0, 1.0, 0.0, 0.0, 0.0)

    hold_start = enter_duration_sec
    hold_end = total_duration_sec - exit_duration_sec

    if elapsed_sec < hold_start:
        # --- ENTER: Slide Left + Fade In + Scale 0.8 -> 1 + Glow Pulse ---
        t = elapsed_sec / enter_duration_sec if enter_duration_sec > 0 else 1.0
        eased_fade = ease_out_cubic(t)
        eased_scale = ease_out_back(t)

        opacity = eased_fade
        scale = 0.8 + 0.2 * eased_scale
        # Слева направо: начинается смещённой вправо (за кадром) и "въезжает" на место.
        x_offset = SLIDE_DISTANCE_PX * (1 - eased_fade)
        y_offset = 0.0
        glow_intensity = eased_fade

    elif elapsed_sec < hold_end:
        # --- HOLD: Floating ---
        t_hold = elapsed_sec - hold_start
        opacity = 1.0
        scale = 1.0
        x_offset = 0.0
        y_offset = FLOAT_AMPLITUDE_PX * sine_wave(t_hold, FLOAT_FREQUENCY_HZ)
        # Glow продолжает мягко пульсировать во время удержания — "дорогой" вид.
        glow_intensity = 0.7 + 0.3 * (0.5 + 0.5 * sine_wave(t_hold, GLOW_PULSE_FREQUENCY_HZ))

    else:
        # --- EXIT: Fade Out + Slide Right ---
        t_exit = (elapsed_sec - hold_end) / exit_duration_sec if exit_duration_sec > 0 else 1.0
        t_exit = clamp01(t_exit)
        eased_exit = ease_out_cubic(t_exit)

        opacity = 1.0 - eased_exit
        scale = 1.0
        x_offset = -SLIDE_DISTANCE_PX * eased_exit  # уезжает вправо (отрицательный x_offset = сдвиг вправо в системе координат рендера)
        y_offset = 0.0
        glow_intensity = 1.0 - eased_exit

    return BannerAnimationState(
        visible=True,
        opacity=clamp01(opacity),
        scale=max(0.01, scale),
        x_offset_px=x_offset,
        y_offset_px=y_offset,
        glow_intensity=clamp01(glow_intensity),
    )


def render_banner_on_frame(
    frame,
    state: BannerAnimationState,
    rect,
    text_lines: list[str],
    font_size: int = 40,
):
    """Композитит плашку (glassmorphism-панель + текст) поверх кадра с учётом
    текущего анимационного состояния (opacity/scale/offset/glow).

    Вынесено сюда, а не в effects/, потому что это единственное место, где
    статический glassmorphism-рендер (Шаг 8) встречается с анимацией по
    времени (этот Шаг 9) — держать их раздельно чище (effects не знает про
    время, animation не знает про пиксельную композицию стекла).
    """
    from PIL import Image, ImageDraw, ImageFont

    from effects.font_utils import DEFAULT_FALLBACK_FONT, resolve_font_path
    from effects.glassmorphism_renderer import GlassPanelStyle, render_glass_panel_on_frame

    if not state.visible or state.opacity <= 0.01:
        return frame

    x1, y1, x2, y2 = rect
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = (x2 - x1) * state.scale, (y2 - y1) * state.scale

    shifted_x1 = center_x - width / 2 + state.x_offset_px
    shifted_y1 = center_y - height / 2 + state.y_offset_px
    shifted_rect = (int(shifted_x1), int(shifted_y1), int(shifted_x1 + width), int(shifted_y1 + height))

    style = GlassPanelStyle(
        fill_rgba=(255, 255, 255, int(30 * state.opacity)),
        border_rgba=(255, 255, 255, int(70 * state.opacity)),
        shadow_rgba=(0, 0, 0, int(140 * state.opacity)),
        glow_rgba=(124, 92, 255, int(90 * state.glow_intensity)),
    )
    panel_frame = render_glass_panel_on_frame(frame, shifted_rect, style)

    # Текст плашки — рисуется поверх панели, с тем же альфа-множителем opacity.
    # ЧЕСТНОЕ ОГРАНИЧЕНИЕ (пойманное визуальным тестом): в отличие от обложки
    # (effects/cover_generator.py), здесь эмодзи не рендерятся отдельным
    # цветным шрифтом построчно-инлайн — это потребовало бы полноценного
    # текстового шейпинга смешанных шрифтов на маленьком кегле. Пока эмодзи
    # аккуратно вырезаются из строк плашки (не показываем "битые" tofu-глифы),
    # а не рендерим как попало.
    from effects.font_utils import is_emoji

    text_layer = Image.new("RGBA", panel_frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    clean_lines = ["".join(ch for ch in line if not is_emoji(ch)).strip() for line in text_lines]
    joined_text = " ".join(clean_lines)
    font_path = resolve_font_path(joined_text, [], DEFAULT_FALLBACK_FONT)
    font = ImageFont.truetype(font_path, font_size)

    line_spacing = int(font_size * 0.3)
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in clean_lines]
    total_height = sum(line_heights) + line_spacing * max(0, len(clean_lines) - 1)
    y_cursor = shifted_rect[1] + ((shifted_rect[3] - shifted_rect[1]) - total_height) // 2

    for line, line_height in zip(clean_lines, line_heights):
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x = shifted_rect[0] + ((shifted_rect[2] - shifted_rect[0]) - line_width) // 2
        alpha = int(255 * state.opacity)
        draw.text((x, y_cursor), line, font=font, fill=(255, 255, 255, alpha))
        y_cursor += line_height + line_spacing

    return Image.alpha_composite(panel_frame, text_layer)
