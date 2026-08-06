"""Анимация логотипа (Функция 13: "Логотип должен быть анимирован").

Простой, но настоящий fade+scale-in за первые LOGO_INTRO_DURATION_SEC секунд
видео, дальше логотип статичен в углу (постоянное присутствие бренда не
должно отвлекать в течение всего ролика — анимация уместна только на входе).
"""

from __future__ import annotations

from dataclasses import dataclass

from animation.easing import clamp01, ease_out_back, ease_out_cubic

LOGO_INTRO_DURATION_SEC = 0.5


@dataclass(frozen=True, slots=True)
class LogoAnimationState:
    opacity: float
    scale: float


def compute_logo_animation_state(
    elapsed_sec: float, intro_duration_sec: float = LOGO_INTRO_DURATION_SEC
) -> LogoAnimationState:
    if elapsed_sec <= 0:
        return LogoAnimationState(opacity=0.0, scale=0.7)
    if elapsed_sec >= intro_duration_sec:
        return LogoAnimationState(opacity=1.0, scale=1.0)

    t = elapsed_sec / intro_duration_sec
    return LogoAnimationState(
        opacity=clamp01(ease_out_cubic(t)),
        scale=0.7 + 0.3 * ease_out_back(t),
    )
