"""Стандартные easing-функции (t: 0..1 -> 0..1), общие для всех аниматоров
проекта — чтобы "Slide Left", "Fade In", "Scale", "Bounce" и т.д. использовали
одинаковый визуальный язык движения, а не разную математику в каждом модуле.
"""

from __future__ import annotations

import math


def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def linear(t: float) -> float:
    return clamp01(t)


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t: float) -> float:
    t = clamp01(t)
    return t ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    if t < 0.5:
        return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Лёгкий "перелёт" за целевое значение перед устаканиванием — то самое
    ощущение "пружинки", которое отличает Scale-анимацию UI-элементов
    (Функция: "Scale 0.8 -> 1") от плоского линейного увеличения.
    """
    t = clamp01(t)
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic_bounce(t: float) -> float:
    """Настоящий "bounce" (Функция: "Subscribe — Bounce Animation") —
    затухающая синусоида поверх экспоненциального спада, не просто overshoot.
    """
    t = clamp01(t)
    if t in (0.0, 1.0):
        return t
    period = 0.3
    return 2 ** (-10 * t) * math.sin((t - period / 4) * (2 * math.pi) / period) + 1


def sine_wave(t: float, frequency_hz: float = 1.0, phase: float = 0.0) -> float:
    """-1..1 синусоида — используется для Floating-анимации (плавное
    покачивание плашки во время "удержания" между входом и выходом)."""
    return math.sin(2 * math.pi * frequency_hz * t + phase)
