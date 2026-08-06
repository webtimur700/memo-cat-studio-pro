"""Единый источник цветов темы.

QSS не умеет rgba()-переменные так же гибко, как код, поэтому цвета для
кастомной отрисовки (glass_panel.py, viral_score_badge.py — QPainter) берутся
отсюда же, а не дублируются магическими числами по всему UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor


@dataclass(frozen=True)
class Palette:
    _background: QColor | None = None
    _surface: QColor | None = None
    _surface_glass: QColor | None = None
    _border_glass: QColor | None = None
    _text_primary: QColor | None = None
    _text_secondary: QColor | None = None
    _accent: QColor | None = None
    _accent_glow: QColor | None = None
    _score_low: QColor | None = None
    _score_mid: QColor | None = None
    _score_high: QColor | None = None
    _danger: QColor | None = None
    _success: QColor | None = None

    @property
    def background(self) -> QColor:
        return self._background if self._background is not None else QColor(18, 18, 22)

    @property
    def surface(self) -> QColor:
        return self._surface if self._surface is not None else QColor(28, 28, 34)

    @property
    def surface_glass(self) -> QColor:
        return self._surface_glass if self._surface_glass is not None else QColor(255, 255, 255, 18)

    @property
    def border_glass(self) -> QColor:
        return self._border_glass if self._border_glass is not None else QColor(255, 255, 255, 28)

    @property
    def text_primary(self) -> QColor:
        return self._text_primary if self._text_primary is not None else QColor(235, 235, 240)

    @property
    def text_secondary(self) -> QColor:
        return self._text_secondary if self._text_secondary is not None else QColor(150, 150, 160)

    @property
    def accent(self) -> QColor:
        return self._accent if self._accent is not None else QColor(124, 92, 255)

    @property
    def accent_glow(self) -> QColor:
        return self._accent_glow if self._accent_glow is not None else QColor(124, 92, 255, 90)

    @property
    def score_low(self) -> QColor:
        return self._score_low if self._score_low is not None else QColor(230, 90, 90)

    @property
    def score_mid(self) -> QColor:
        return self._score_mid if self._score_mid is not None else QColor(230, 190, 70)

    @property
    def score_high(self) -> QColor:
        return self._score_high if self._score_high is not None else QColor(90, 220, 140)

    @property
    def danger(self) -> QColor:
        return self._danger if self._danger is not None else QColor(230, 90, 90)

    @property
    def success(self) -> QColor:
        return self._success if self._success is not None else QColor(90, 220, 140)


PALETTE = Palette()


def score_color(score: int) -> QColor:
    """Возвращает цвет badge'а в зависимости от Viral Score (0-100)."""
    if score < 40:
        return PALETTE.score_low
    if score < 70:
        return PALETTE.score_mid
    return PALETTE.score_high
