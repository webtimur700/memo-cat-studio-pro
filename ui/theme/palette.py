from dataclasses import dataclass, field
from PySide6.QtGui import QColor

@dataclass(frozen=True, slots=True)
class Palette:
    """Цветовая палитра приложения (Dark Mode по умолчанию)."""
    background: QColor = field(default_factory=lambda: QColor(18, 18, 22))
    surface: QColor = field(default_factory=lambda: QColor(28, 28, 34))
    surface_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 18))
    
    text_primary: QColor = field(default_factory=lambda: QColor(235, 235, 245))
    text_secondary: QColor = field(default_factory=lambda: QColor(160, 160, 170))
    text_muted: QColor = field(default_factory=lambda: QColor(90, 90, 100))
    
    accent: QColor = field(default_factory=lambda: QColor(114, 137, 218))
    accent_hover: QColor = field(default_factory=lambda: QColor(130, 150, 230))
    accent_pressed: QColor = field(default_factory=lambda: QColor(90, 110, 190))
    
    success: QColor = field(default_factory=lambda: QColor(80, 200, 120))
    warning: QColor = field(default_factory=lambda: QColor(255, 190, 80))
    error: QColor = field(default_factory=lambda: QColor(255, 100, 100))
    
    border: QColor = field(default_factory=lambda: QColor(60, 60, 70))
    border_focus: QColor = field(default_factory=lambda: QColor(114, 137, 218))
    
    shadow: QColor = field(default_factory=lambda: QColor(0, 0, 0, 100))
    overlay: QColor = field(default_factory=lambda: QColor(0, 0, 0, 140))

# Глобальный экземпляр палитры
PALETTE = Palette()
