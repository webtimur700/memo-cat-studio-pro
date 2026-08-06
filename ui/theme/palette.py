"""Единый источник цветов темы.
     2	
     3	QSS не умеет rgba()-переменные так же гибко, как код, поэтому цвета для
     4	кастомной отрисовки (glass_panel.py, viral_score_badge.py — QPainter) берутся
     5	отсюда же, а не дублируются магическими числами по всему UI.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	from dataclasses import dataclass, field
    11	
    12	from PySide6.QtGui import QColor
    13	
    14	
    15	@dataclass(frozen=True, slots=True)
    16	class Palette:
    17	    background: QColor = field(default_factory=lambda: QColor(18, 18, 22))
    18	    surface: QColor = field(default_factory=lambda: QColor(28, 28, 34))
    19	    surface_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 18))
    20	    border_glass: QColor = field(default_factory=lambda: QColor(255, 255, 255, 28))
    21	
    22	    text_primary: QColor = field(default_factory=lambda: QColor(235, 235, 240))
    23	    text_secondary: QColor = field(default_factory=lambda: QColor(150, 150, 160))
    24	
    25	    accent: QColor = field(default_factory=lambda: QColor(124, 92, 255))      # фиолетовый акцент — фирменный цвет Memo Cat
    26	    accent_glow: QColor = field(default_factory=lambda: QColor(124, 92, 255, 90))
    27	
    28	    score_low: QColor = field(default_factory=lambda: QColor(230, 90, 90))     # viral score < 40
    29	    score_mid: QColor = field(default_factory=lambda: QColor(230, 190, 70))    # 40-70
    30	    score_high: QColor = field(default_factory=lambda: QColor(90, 220, 140))   # > 70
    31	
    32	    danger: QColor = field(default_factory=lambda: QColor(230, 90, 90))
    33	    success: QColor = field(default_factory=lambda: QColor(90, 220, 140))
    34	
    35	
    36	PALETTE = Palette()
    37	
    38	
    39	def score_color(score: int) -> QColor:
    40	    """Возвращает цвет badge'а в зависимости от Viral Score (0-100)."""
    41	    if score < 40:
    42	        return PALETTE.score_low
    43	    if score < 70:
    44	        return PALETTE.score_mid
    45	    return PALETTE.score_high
    46	
