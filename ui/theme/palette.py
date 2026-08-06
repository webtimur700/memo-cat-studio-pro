"""Единый источник цветов темы.

QSS не умеет rgba()-переменные так же гибко, как код, поэтому цвета для
кастомной отрисовки (glass_panel.py, viral_score_badge.py — QPainter) берутся
отсюда же, а не дублируются магическими числами по всему UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class Palette:
    background: QColor = field(default_factory=lambda: QColor(18, 18, 22))
    surface: QColor = field(default_factory=lambda: QColor(28, 28, 34))
    surface_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 18))
    border_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 28))

    text_primary: QColor = field(default_factory=lambda: QColor(235, 235, 240))
    text_secondary: QColor = field(default_factory=lambda: QColor(150, 150, 160))

    accent: QColor = field(default_factory=lambda: QColor(124, 92, 255))
    accent_glow: QColor = field(default_factory=lambda: QColor(124, 92, 255, 90))

    score_low: QColor = field(default_factory=lambda: QColor(230, 90, 90))
    score_mid: QColor = field(default_factory=lambda: QColor(230, 190, 70))
    score_high: QColor = field(default_factory=lambda: QColor(90, 220, 140))

    danger: QColor = field(default_factory=lambda: QColor(230, 90, 90))
    success: QColor = field(default_factory=lambda: QColor(90, 220, 140))


PALETTE = Palette()


def score_color(score: int) -> QColor:
    """Возвращает цвет badge'а в зависимости от Viral Score (0-100)."""
    if score < 40:
        return PALETTE.score_low
    if score < 70:
        return PALETTE.score_mid
    return PALETTE.score_high
