"""Bounce-анимация кнопки Subscribe (Функция 14). Появляется один раз в
начале ролика с "пружинным" эффектом, затем периодически слегка подпрыгивает
каждые REPEAT_INTERVAL_SEC — привлекает внимание, не будучи навязчивой
анимацией на каждом кадре.
"""

from __future__ import annotations

from animation.easing import clamp01, ease_out_elastic_bounce

INTRO_BOUNCE_DURATION_SEC = 0.6
REPEAT_INTERVAL_SEC = 4.0
REPEAT_BOUNCE_DURATION_SEC = 0.4
REPEAT_BOUNCE_SCALE_AMPLITUDE = 0.12  # слабее, чем интро-bounce — не отвлекать повторами


def compute_subscribe_button_scale(elapsed_sec: float) -> float:
    if elapsed_sec < 0:
        return 0.0  # кнопка ещё не должна быть в кадре

    if elapsed_sec < INTRO_BOUNCE_DURATION_SEC:
        t = elapsed_sec / INTRO_BOUNCE_DURATION_SEC
        return ease_out_elastic_bounce(t)

    # После интро — периодические лёгкие "пульсы" каждые REPEAT_INTERVAL_SEC,
    # а не постоянная анимация (постоянный bounce утомляет зрителя за 15-60 секунд ролика).
    time_since_last_cycle_start = (elapsed_sec - INTRO_BOUNCE_DURATION_SEC) % REPEAT_INTERVAL_SEC
    if time_since_last_cycle_start >= REPEAT_BOUNCE_DURATION_SEC:
        return 1.0

    t = time_since_last_cycle_start / REPEAT_BOUNCE_DURATION_SEC
    bounce = ease_out_elastic_bounce(t)
    # Малая амплитуда вокруг 1.0, а не полный bounce от 0 — кнопка уже на месте.
    return 1.0 + (bounce - 1.0) * REPEAT_BOUNCE_SCALE_AMPLITUDE
