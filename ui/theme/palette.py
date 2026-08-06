"""Единый источник цветов темы.

QSS не умеет rgba()-переменные так же гибко, как код, поэтому цвета для
кастомной отрисовки (glass_panel.py, viral_score_badge.py — QPainter) берутся
отсюда же, а не дублируются магическими числами по всему UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class Palette:
    background: QColor = QColor(18, 18, 22)
    surface: QColor = QColor(28, 28, 34)
    surface_glass: QColor = QColor(255, 255, 255, 18)
    border_glass: QColor = QColor(255, 255, 255, 28)

    text_primary: QColor = QColor(235, 235, 240)
    text_secondary: QColor = QColor(150, 150, 160)

    accent: QColor = QColor(124, 92, 255)      # фиолетовый акцент — фирменный цвет Memo Cat
    accent_glow: QColor = QColor(124, 92, 255, 90)

    score_low: QColor = QColor(230, 90, 90)     # viral score < 40
    score_mid: QColor = QColor(230, 190, 70)    # 40-70
    score_high: QColor = QColor(90, 220, 140)   # > 70

    danger: QColor = QColor(230, 90, 90)
    success: QColor = QColor(90, 220, 140)


PALETTE = Palette()


def score_color(score: int) -> QColor:
    """Возвращает цвет badge'а в зависимости от Viral Score (0-100)."""
    if score < 40:
        return PALETTE.score_low
    if score < 70:
        return PALETTE.score_mid
    return PALETTE.score_high
