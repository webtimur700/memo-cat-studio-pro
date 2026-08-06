from dataclasses import dataclass, field
from PySide6.QtGui import QColor

dataclass(frozen=True, slots=True)
class Palette:
    background: QColor = field(default_factory=lambda: QColor(18, 18, 22))
    surface: QColor = field(default_factory=lambda: QColor(28, 28, 34))
    surface_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 18))
    
    text_primary: QColor = field(default_factory=lambda: QColor(255, 255, 255))
    text_secondary: QColor = field(default_factory=lambda: QColor(160, 160, 160))
    text_muted: QColor = field(default_factory=lambda: QColor(90, 90, 90))
    
    accent: QColor = field(default_factory=lambda: QColor(0, 191, 255))
    accent_glow: QColor = field(default_factory=lambda: QColor(0, 191, 255, 60))
    success: QColor = field(default_factory=lambda: QColor(0, 255, 127))
    warning: QColor = field(default_factory=lambda: QColor(255, 215, 0))
    error: QColor = field(default_factory=lambda: QColor(255, 69, 0))
    
    border: QColor = field(default_factory=lambda: QColor(255, 255, 255, 30))
    overlay: QColor = field(default_factory=lambda: QColor(0, 0, 0, 140))

PALETTE = Palette()


def score_color(score: float) -> QColor:
    """Возвращает цвет в зависимости от уверенности (0.0 - 1.0)."""
    if score >= 0.8:
        return PALETTE.success
    if score >= 0.5:
        return PALETTE.warning
    return PALETTE.error

