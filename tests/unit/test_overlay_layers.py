"""Быстрые слои overlay (effects/overlay_layers.py) обязаны давать те же пиксели, что и прежний покадровый рендер."""

import subprocess

import numpy as np
from PIL import Image

from animation.promo_banner_animator import compute_banner_animation_state, render_banner_on_frame
from core.entities.settings import UserSettings
from effects.branding_overlay import BrandingOverlay, resolve_logo_path
from effects.overlay_layers import banner_item, render_overlay_layers
from effects.safe_zone import SafeZone

FRAME = (1080, 1920)
RECT = (90, 1300, 990, 1520)
LINES = ("💼 100zarplat.ru", "Подбор вакансий")


def _old_banner(elapsed: float) -> np.ndarray:
    state = compute_banner_animation_state(elapsed, 5.0)
    blank = Image.new("RGBA", FRAME, (0, 0, 0, 0))
    return np.asarray(render_banner_on_frame(blank, state, RECT, list(LINES)), dtype=np.int16)


def _new_banner(elapsed: float) -> np.ndarray:
    state = compute_banner_animation_state(elapsed, 5.0)
    item = banner_item(state, RECT, LINES)
    canvas = Image.new("RGBA", FRAME, (0, 0, 0, 0))
    canvas.alpha_composite(item.sprite, item.xy)
    return np.asarray(canvas, dtype=np.int16)


def test_banner_patch_matches_full_frame_render():
    """Вход, удержание (плавающее смещение и пульс свечения) и выход: пиксели совпадают с прежним рендером."""
    for elapsed in (0.05, 0.2, 0.44, 0.9, 2.3, 3.1, 4.7, 4.9):
        diff = np.abs(_old_banner(elapsed) - _new_banner(elapsed))
        assert diff.max() <= 1, f"t={elapsed}: максимальное расхождение {diff.max()}"


def test_layers_cover_only_visible_frames_and_stay_small(tmp_path):
    zone = SafeZone.from_settings(FRAME, UserSettings().safe_zone)
    branding = BrandingOverlay.build(FRAME, resolve_logo_path(tmp_path), "top_right", True, zone=zone)
    layers = {
        layer.name: layer
        for layer in render_overlay_layers(tmp_path / "out", 10.0, FRAME, 30, RECT, list(LINES), 4.0, 5.0, branding)
    }
    assert set(layers) == {"banner", "logo", "subscribe"}
    assert 120 <= layers["banner"].start_frame <= 122 and 145 <= layers["banner"].frame_count <= 152      # 4.0 .. 9.0 с
    assert 30 <= layers["subscribe"].start_frame <= 32 and 175 <= layers["subscribe"].frame_count <= 182  # 1.0 .. 7.0 с
    assert layers["logo"].start_frame <= 1 and layers["logo"].frame_count >= 298
    for layer in layers.values():
        assert layer.path.stat().st_size == layer.width * layer.height * 4 * layer.frame_count
        assert layer.width < FRAME[0] * 1.3 and layer.width * layer.height < FRAME[0] * FRAME[1] / 3   # не полный кадр
    assert layers["logo"].width * layers["logo"].height < 60_000


def test_no_layers_without_banner_and_branding(tmp_path):
    assert render_overlay_layers(tmp_path / "o", 3.0, FRAME, 30, None, [], 4.0, 5.0, None) == []


def test_overlay_appears_in_video_at_the_right_time(tmp_path):
    """Сквозной ffmpeg: плашка на чёрном фоне не видна до 4.0 с и видна после; логотип виден с самого начала."""
    zone = SafeZone.from_settings(FRAME, UserSettings().safe_zone)
    branding = BrandingOverlay.build(FRAME, resolve_logo_path(tmp_path), "top_right", False, zone=zone)
    layers = render_overlay_layers(tmp_path / "l", 6.0, FRAME, 30, RECT, list(LINES), 4.0, 1.5, branding)
    from export.export_service import _layer_input_args, _overlay_graph

    graph, label = _overlay_graph("null", layers, 1, "")
    out = tmp_path / "o.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=6",
         *[a for layer in layers for a in _layer_input_args(layer, 30)],
         "-filter_complex", graph, "-map", f"[{label}]", "-t", "6", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )

    def luma(t: float, box: tuple[int, int, int, int]) -> float:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-vf", "format=gray",
                              "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        frame = np.frombuffer(raw, np.uint8).reshape(1920, 1080)
        return float(frame[box[1]:box[3], box[0]:box[2]].mean())

    assert luma(3.0, RECT) < 2 and luma(4.8, RECT) > 5 and luma(5.9, RECT) < 2     # плашка: 4.0-5.5 с
    logo = layers[1]
    assert luma(2.0, (logo.x, logo.y, logo.x + logo.width, logo.y + logo.height)) > 10
