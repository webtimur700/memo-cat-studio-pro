"""Логотип и кнопка Subscribe для прозрачного overlay-ролика.

Логотип: если есть assets/logo/logo.png — берётся он (пользовательский файл
всегда важнее); иначе используется/создаётся простой PNG "Memo Cat"
(assets/logo/default_logo.png). Анимации считают animation/logo_animator.py
(fade+scale на входе) и animation/subscribe_button_animator.py (bounce),
здесь — только спрайты, позиционирование и композитинг на кадр.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from animation.logo_animator import compute_logo_animation_state
from animation.subscribe_button_animator import compute_subscribe_button_scale
from effects.font_utils import DEFAULT_FALLBACK_FONT, preferred_title_fonts, resolve_font_path

USER_LOGO_NAME = "logo.png"
DEFAULT_LOGO_NAME = "default_logo.png"

LOGO_WIDTH_RATIO = 0.19        # ширина логотипа от ширины кадра
EDGE_MARGIN_PX = 48
TOP_SAFE_MARGIN_PX = 120       # верх Shorts перекрывает интерфейс плеера
BOTTOM_SAFE_MARGIN_PX = 260

SUBSCRIBE_APPEAR_AT_SEC = 1.0
SUBSCRIBE_VISIBLE_SEC = 6.0
SUBSCRIBE_EXIT_SEC = 0.25
SUBSCRIBE_TEXT = "ПОДПИСАТЬСЯ"


def create_default_logo(path: Path, size: tuple[int, int] = (640, 240)) -> Path:
    """Простой логотип: фиолетовая плашка, "ушки" и белый текст "Memo Cat"."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    accent = (124, 92, 255, 255)

    ear_h = int(height * 0.32)
    body_top = ear_h - 8
    for x0 in (int(width * 0.10), int(width * 0.68)):
        draw.polygon(
            [(x0, body_top + 20), (x0 + int(width * 0.09), 0), (x0 + int(width * 0.22), body_top + 20)],
            fill=accent,
        )
    draw.rounded_rectangle((0, body_top, width - 1, height - 1), radius=int(height * 0.28), fill=accent)

    font_path = resolve_font_path("Memo Cat", preferred_title_fonts(), DEFAULT_FALLBACK_FONT)
    font_size = int(height * 0.42)
    font = ImageFont.truetype(font_path, font_size)
    while draw.textlength("Memo Cat", font=font) > width * 0.86 and font_size > 12:
        font_size -= 4
        font = ImageFont.truetype(font_path, font_size)
    text_w = draw.textlength("Memo Cat", font=font)
    draw.text(
        ((width - text_w) / 2, body_top + (height - body_top) / 2),
        "Memo Cat", font=font, fill=(255, 255, 255, 255), anchor="lm",
        stroke_width=2, stroke_fill=(40, 20, 110, 255),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def resolve_logo_path(assets_dir: Path) -> Path | None:
    """assets/logo/logo.png, если есть; иначе default_logo.png (создаётся при отсутствии)."""
    logo_dir = assets_dir / "logo"
    user_logo = logo_dir / USER_LOGO_NAME
    if user_logo.is_file():
        return user_logo
    default_logo = logo_dir / DEFAULT_LOGO_NAME
    if default_logo.is_file():
        return default_logo
    try:
        return create_default_logo(default_logo)
    except Exception as exc:
        logger.warning("Не удалось создать логотип по умолчанию {}: {}", default_logo, exc)
        return None


def load_logo_sprite(logo_path: Path, frame_width: int) -> Image.Image:
    logo = Image.open(logo_path).convert("RGBA")
    target_w = max(1, int(frame_width * LOGO_WIDTH_RATIO))
    target_h = max(1, round(logo.height * target_w / logo.width))
    return logo.resize((target_w, target_h), Image.Resampling.LANCZOS)


def render_subscribe_sprite(frame_width: int) -> Image.Image:
    """Красная "таблетка" с белым текстом и треугольником play."""
    height = max(64, int(frame_width * 0.085))
    font_path = resolve_font_path(SUBSCRIBE_TEXT, preferred_title_fonts(), DEFAULT_FALLBACK_FONT)
    font = ImageFont.truetype(font_path, int(height * 0.42))

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_w = int(probe.textlength(SUBSCRIBE_TEXT, font=font))
    pad = int(height * 0.45)
    icon = int(height * 0.36)
    width = pad + icon + int(height * 0.28) + text_w + pad

    sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=height // 2, fill=(230, 33, 23, 255))
    cy = height // 2
    draw.polygon([(pad, cy - icon // 2), (pad, cy + icon // 2), (pad + icon, cy)], fill=(255, 255, 255, 255))
    draw.text((pad + icon + int(height * 0.28), cy), SUBSCRIBE_TEXT, font=font, fill=(255, 255, 255, 255), anchor="lm")
    return sprite


def logo_xy(position: str, frame_size: tuple[int, int], sprite_size: tuple[int, int]) -> tuple[int, int]:
    """Левый верхний угол логотипа для top_left/top_right/bottom_left/bottom_right/top_center/bottom_center."""
    fw, fh = frame_size
    sw, sh = sprite_size
    vertical, _, horizontal = position.partition("_")
    x = {"left": EDGE_MARGIN_PX, "right": fw - sw - EDGE_MARGIN_PX}.get(horizontal, (fw - sw) // 2)
    y = fh - sh - BOTTOM_SAFE_MARGIN_PX if vertical == "bottom" else TOP_SAFE_MARGIN_PX
    return x, y


def subscribe_xy(logo_position: str, frame_size: tuple[int, int], sprite_size: tuple[int, int]) -> tuple[int, int]:
    """Кнопка — на противоположной от логотипа стороне, на той же высоте: не
    закрывает логотип и не лезет в правую колонку кнопок плеера Shorts."""
    fw, fh = frame_size
    sw, sh = sprite_size
    vertical, _, horizontal = logo_position.partition("_")
    x = fw - sw - EDGE_MARGIN_PX if horizontal == "left" else EDGE_MARGIN_PX
    y = fh - sh - BOTTOM_SAFE_MARGIN_PX if vertical == "bottom" else TOP_SAFE_MARGIN_PX
    return x, y


def _scaled_with_opacity(sprite: Image.Image, scale: float, opacity: float) -> Image.Image | None:
    if scale <= 0.01 or opacity <= 0.01:
        return None
    if abs(scale - 1.0) > 0.005:
        sprite = sprite.resize(
            (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))), Image.Resampling.BILINEAR
        )
    if opacity < 0.995:
        alpha = sprite.getchannel("A").point(lambda v: int(v * opacity))
        sprite = sprite.copy()
        sprite.putalpha(alpha)
    return sprite


def _paste_centered(canvas: Image.Image, sprite: Image.Image, top_left: tuple[int, int], base_size: tuple[int, int]):
    cx, cy = top_left[0] + base_size[0] / 2, top_left[1] + base_size[1] / 2
    canvas.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))


@dataclass
class BrandingOverlay:
    """Спрайты и позиции; draw_on() дорисовывает логотип и Subscribe на кадр в момент t."""

    frame_size: tuple[int, int]
    logo: Image.Image | None
    logo_position: str
    subscribe: Image.Image | None

    @classmethod
    def build(
        cls,
        frame_size: tuple[int, int],
        logo_path: Path | None,
        logo_position: str,
        subscribe_enabled: bool,
    ) -> "BrandingOverlay":
        logo = None
        if logo_path is not None:
            try:
                logo = load_logo_sprite(logo_path, frame_size[0])
            except Exception as exc:
                logger.warning("Логотип {} не загружен: {} — без логотипа", logo_path, exc)
        subscribe = render_subscribe_sprite(frame_size[0]) if subscribe_enabled else None
        return cls(frame_size, logo, logo_position, subscribe)

    @property
    def is_empty(self) -> bool:
        return self.logo is None and self.subscribe is None

    def obstacle_rects(self, window_start_sec: float, window_end_sec: float) -> list[tuple[int, int, int, int]]:
        """Прямоугольники (x1, y1, x2, y2) логотипа и кнопки Subscribe, которые
        видны в интервале [window_start_sec, window_end_sec] — для обхода плашкой."""
        rects: list[tuple[int, int, int, int]] = []
        if self.logo is not None:
            x, y = logo_xy(self.logo_position, self.frame_size, self.logo.size)
            rects.append((x, y, x + self.logo.width, y + self.logo.height))
        if self.subscribe is not None:
            visible_from = SUBSCRIBE_APPEAR_AT_SEC
            visible_to = SUBSCRIBE_APPEAR_AT_SEC + SUBSCRIBE_VISIBLE_SEC
            if visible_from < window_end_sec and window_start_sec < visible_to:
                x, y = subscribe_xy(self.logo_position, self.frame_size, self.subscribe.size)
                rects.append((x, y, x + self.subscribe.width, y + self.subscribe.height))
        return rects

    def draw_on(self, canvas: Image.Image, t: float) -> bool:
        """Рисует на canvas (RGBA, изменяется на месте). True, если что-то нарисовано."""
        drawn = False
        if self.logo is not None:
            state = compute_logo_animation_state(t)
            sprite = _scaled_with_opacity(self.logo, state.scale, state.opacity)
            if sprite is not None:
                _paste_centered(canvas, sprite, logo_xy(self.logo_position, self.frame_size, self.logo.size), self.logo.size)
                drawn = True

        if self.subscribe is not None:
            elapsed = t - SUBSCRIBE_APPEAR_AT_SEC
            scale = compute_subscribe_button_scale(elapsed)
            remaining = SUBSCRIBE_VISIBLE_SEC - elapsed
            if elapsed >= 0 and remaining > 0:
                if remaining < SUBSCRIBE_EXIT_SEC:
                    scale *= remaining / SUBSCRIBE_EXIT_SEC
                sprite = _scaled_with_opacity(self.subscribe, scale, min(1.0, elapsed / 0.15 + 0.01))
                if sprite is not None:
                    xy = subscribe_xy(self.logo_position, self.frame_size, self.subscribe.size)
                    _paste_centered(canvas, sprite, xy, self.subscribe.size)
                    drawn = True
        return drawn
