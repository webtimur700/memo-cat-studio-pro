from animation.easing import clamp01, ease_in_cubic, ease_out_back, ease_out_cubic, sine_wave
from animation.logo_animator import compute_logo_animation_state
from animation.promo_banner_animator import ENTER_DURATION_SEC, compute_banner_animation_state
from animation.subscribe_button_animator import INTRO_BOUNCE_DURATION_SEC, compute_subscribe_button_scale


def test_easing_boundary_conditions():
    assert ease_out_cubic(0) == 0 and abs(ease_out_cubic(1) - 1) < 1e-9
    assert ease_in_cubic(0) == 0 and abs(ease_in_cubic(1) - 1) < 1e-9
    assert abs(ease_out_back(0)) < 1e-9 and abs(ease_out_back(1) - 1) < 1e-9
    assert clamp01(-0.5) == 0.0 and clamp01(1.5) == 1.0


def test_ease_out_back_overshoots():
    mid_values = [ease_out_back(t) for t in (0.5, 0.6, 0.7, 0.8)]
    assert max(mid_values) > 1.0


def test_sine_wave_range():
    assert abs(sine_wave(0.0)) < 1e-9
    assert abs(sine_wave(0.25) - 1.0) < 1e-9
    assert abs(sine_wave(0.5)) < 1e-9


def test_banner_animation_full_lifecycle():
    total = 5.0
    before = compute_banner_animation_state(-0.1, total)
    after = compute_banner_animation_state(5.1, total)
    assert before.visible is False and after.visible is False

    start = compute_banner_animation_state(0.0, total)
    assert start.opacity < 0.05
    assert abs(start.scale - 0.8) < 0.05

    settled = compute_banner_animation_state(ENTER_DURATION_SEC, total)
    assert settled.opacity > 0.99
    assert abs(settled.x_offset_px) < 1.0

    hold = compute_banner_animation_state(2.5, total)
    assert hold.opacity == 1.0
    assert abs(hold.y_offset_px) <= 6.0

    end = compute_banner_animation_state(total, total)
    assert end.opacity < 0.05


def test_logo_animation():
    assert compute_logo_animation_state(0.0).opacity == 0.0
    end_state = compute_logo_animation_state(0.5)
    assert end_state.opacity == 1.0 and end_state.scale == 1.0


def test_subscribe_bounce():
    assert compute_subscribe_button_scale(-0.1) == 0.0
    settled = compute_subscribe_button_scale(INTRO_BOUNCE_DURATION_SEC + 0.05)
    assert abs(settled - 1.0) < 0.15
