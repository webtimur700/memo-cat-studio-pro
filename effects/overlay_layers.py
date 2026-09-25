"""Быстрый рендер анимированных оверлеев (плашка, логотип, Subscribe) в «слои» для ffmpeg.

Раньше на КАЖДЫЙ кадр клипа Pillow собирал полный RGBA-кадр 1080x1920, PNG-файл, и всё это
кодировалось в qtrle/MOV (~30 с на клип). Здесь тот же результат получается дешевле:

* каждый элемент — свой слой в рамке, равной объединению его положений (у логотипа ~200x80 px,
  а не весь кадр), и только на те кадры, где он виден;
* картинки не пересчитываются: спрайт/плашка кэшируется по параметрам анимации (после
  вступления логотип и Subscribe неподвижны, «удержание» плашки повторяет ~30 разных стекол);
* плашка считается только в своей рамке (а не на кадре целиком), без бесполезного размытия
  прозрачного фона (под плашкой в оверлее всегда пусто: размытие нуля — ноль);
* слой пишется как rawvideo RGBA без сжатия, ffmpeg накладывает его через `-itsoffset` и
  `overlay=eof_action=pass` прямо в проходе кодирования, промежуточного qtrle-ролика нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from animation.promo_banner_animator import BannerAnimationState, compute_banner_animation_state
from effects.banner_text import draw_banner_text
from effects.branding_overlay import BrandingOverlay, OverlayItem
from effects.glassmorphism_renderer import GlassPanelStyle

# Запас вокруг плашки под размытие тени (24) и свечения (20 + линия 6) — за ~3 сигмы альфа уже нулевая.
PATCH_MARGIN_PX = 110
BANNER_FONT_SIZE = 40
_CACHE_LIMIT = 48


@dataclass(frozen=True)
class OverlayLayer:
    """Один слой: rawvideo RGBA `width`x`height` в файле, накладывается в (x, y) с кадра `start_frame`."""

    name: str
    path: Path
    x: int
    y: int
    width: int
    height: int
    start_frame: int
    frame_count: int


# ----------------------------------------------------------------------
# плашка
# ----------------------------------------------------------------------
# Кэши живут между клипами: у клипов одного размера плашки «въезд», «удержание» и «уход» совпадают.
_layer_cache: dict[tuple, Image.Image] = {}
_patch_cache: dict[tuple, Image.Image] = {}


def _cached(cache: dict[tuple, Image.Image], key: tuple, build) -> Image.Image:
    image = cache.get(key)
    if image is None:
        if len(cache) >= _CACHE_LIMIT:
            cache.pop(next(iter(cache)))
        image = cache[key] = build()
    return image


def _blurred_shape(size: tuple[int, int], draw_shape, radius: int) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_shape(ImageDraw.Draw(layer))
    return layer.filter(ImageFilter.GaussianBlur(radius))


def banner_patch(width: int, height: int, style: GlassPanelStyle, text_lines: tuple[str, ...], text_opacity: float) -> Image.Image:
    """Плашка (тень, свечение, панель, текст) в собственных координатах: панель лежит в (M, M)-(M+w, M+h).

    Те же шаги и та же последовательность наложения, что у render_glass_panel_on_frame + render_banner_on_frame,
    кроме размытия фона под плашкой: в прозрачном оверлее фон пуст, размытие пустоты — пустота."""
    m = PATCH_MARGIN_PX
    size = (width + 2 * m + 1, height + 2 * m + 1)
    rect = (m, m, m + width, m + height)
    sx, sy = style.shadow_offset

    shadow = _cached(_layer_cache, ("shadow", width, height, style.shadow_rgba), lambda: _blurred_shape(
        size,
        lambda d: d.rounded_rectangle((m + sx, m + sy, m + width + sx, m + height + sy), radius=style.corner_radius, fill=style.shadow_rgba),
        style.shadow_blur_radius,
    ))
    glow = _cached(_layer_cache, ("glow", width, height, style.glow_rgba), lambda: _blurred_shape(
        size,
        lambda d: d.rounded_rectangle(rect, radius=style.corner_radius, outline=style.glow_rgba, width=6),
        style.glow_radius,
    ))

    def build_panel() -> Image.Image:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(rect, radius=style.corner_radius, fill=style.fill_rgba)
        draw.rounded_rectangle(rect, radius=style.corner_radius, outline=style.border_rgba, width=style.border_width)
        return layer

    panel = _cached(_layer_cache, ("panel", width, height, style.fill_rgba, style.border_rgba, style.border_width), build_panel)

    def build_text() -> Image.Image:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw_banner_text(layer, rect, list(text_lines), text_opacity, start_font_size=BANNER_FONT_SIZE)
        return layer

    text = _cached(_layer_cache, ("text", width, height, text_lines, text_opacity), build_text)

    result = Image.alpha_composite(shadow, glow)
    result = Image.alpha_composite(result, panel)
    return Image.alpha_composite(result, text)


def banner_item(
    state: BannerAnimationState, rect: tuple[int, int, int, int], text_lines: tuple[str, ...]
) -> OverlayItem | None:
    """Плашка для состояния анимации: спрайт-патч и его угол на кадре (None — плашки нет)."""
    if not state.visible or state.opacity <= 0.01:
        return None
    x1, y1, x2, y2 = rect
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = (x2 - x1) * state.scale, (y2 - y1) * state.scale
    sx1 = center_x - width / 2 + state.x_offset_px
    sy1 = center_y - height / 2 + state.y_offset_px
    shifted = (int(sx1), int(sy1), int(sx1 + width), int(sy1 + height))
    w, h = shifted[2] - shifted[0], shifted[3] - shifted[1]
    style = GlassPanelStyle(
        fill_rgba=(255, 255, 255, int(30 * state.opacity)),
        border_rgba=(255, 255, 255, int(70 * state.opacity)),
        shadow_rgba=(0, 0, 0, int(140 * state.opacity)),
        glow_rgba=(124, 92, 255, int(90 * state.glow_intensity)),
    )
    key = ("banner", w, h, style.fill_rgba, style.border_rgba, style.shadow_rgba, style.glow_rgba, state.opacity, text_lines)
    patch = _cached(_patch_cache, key, lambda: banner_patch(w, h, style, text_lines, state.opacity))
    return OverlayItem(key, patch, (shifted[0] - PATCH_MARGIN_PX, shifted[1] - PATCH_MARGIN_PX))


# ----------------------------------------------------------------------
# слои
# ----------------------------------------------------------------------
def _clip_box(item: OverlayItem, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = item.xy
    return max(0, x), max(0, y), min(frame_size[0], x + item.sprite.width), min(frame_size[1], y + item.sprite.height)


def _write_layer(
    name: str, items: list[OverlayItem | None], frame_size: tuple[int, int], directory: Path
) -> OverlayLayer | None:
    """Пишет слой из покадровых элементов (None — на кадре элемента нет). Кадры до первого и после
    последнего непустого не пишутся: ffmpeg накладывает слой со смещением и без «залипания» на последнем кадре."""
    present = [i for i, item in enumerate(items) if item is not None]
    if not present:
        return None
    first, last = present[0], present[-1]
    boxes = [_clip_box(items[i], frame_size) for i in present]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return None
    bx1, by1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    bx2, by2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    width, height = bx2 - bx1, by2 - by1

    path = directory / f"layer_{name}.rgba"
    empty = bytes(width * height * 4)
    previous_id: tuple | None = None
    frame_bytes = empty
    with path.open("wb") as out:
        for item in items[first:last + 1]:
            if item is None:
                previous_id, frame_bytes = None, empty
            elif (item.key, item.xy) != previous_id:
                canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                x, y = item.xy[0] - bx1, item.xy[1] - by1
                sprite = item.sprite
                if x < 0 or y < 0:   # alpha_composite не принимает отрицательный сдвиг: отрезаем выступающее
                    sprite = sprite.crop((max(0, -x), max(0, -y), sprite.width, sprite.height))
                    x, y = max(0, x), max(0, y)
                canvas.alpha_composite(sprite, (x, y))
                previous_id, frame_bytes = (item.key, item.xy), canvas.tobytes()
            out.write(frame_bytes)
    return OverlayLayer(name, path, bx1, by1, width, height, first, last - first + 1)


def render_overlay_layers(
    directory: Path,
    clip_duration_sec: float,
    frame_size: tuple[int, int],
    fps: int,
    banner_rect: tuple[int, int, int, int] | None,
    banner_text_lines: list[str],
    appear_at_sec: float,
    banner_duration_sec: float,
    branding: BrandingOverlay | None,
) -> list[OverlayLayer]:
    """Слои оверлея на весь клип в порядке наложения: плашка, логотип, Subscribe."""
    directory.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(clip_duration_sec * fps))
    lines = tuple(banner_text_lines)
    has_branding = branding is not None and not branding.is_empty

    banner_items: list[OverlayItem | None] = []
    logo_items: list[OverlayItem | None] = []
    subscribe_items: list[OverlayItem | None] = []
    for frame_index in range(total_frames):
        t = frame_index / fps
        if banner_rect is not None:
            state = compute_banner_animation_state(t - appear_at_sec, banner_duration_sec)
            banner_items.append(banner_item(state, banner_rect, lines))
        if has_branding:
            logo_items.append(branding.logo_item(t))
            subscribe_items.append(branding.subscribe_item(t))

    layers = []
    for name, items in (("banner", banner_items), ("logo", logo_items), ("subscribe", subscribe_items)):
        if items:
            layer = _write_layer(name, items, frame_size, directory)
            if layer is not None:
                layers.append(layer)
    return layers
