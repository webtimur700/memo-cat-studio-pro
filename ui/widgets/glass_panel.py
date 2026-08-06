"""Переиспользуемый glassmorphism-виджет.

Используется двояко:
  1. Как обычная панель в UI (карточки настроек, боковые блоки).
  2. Как визуальная референс-реализация для effects/glassmorphism_renderer.py
     (Шаг 8) — там тот же алгоритм (полупрозрачная заливка + мягкая тень +
     скруглённые углы + едва заметная светлая окантовка) переносится в OpenCV/PIL
     для рендера в видео. Держим оба места визуально согласованными.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

from ui.theme.palette import PALETTE


class GlassPanel(QWidget):
    """QWidget с полупрозрачным "стеклянным" фоном, скруглёнными углами
    и мягким свечением (glow) по краю — без обращения к системному QSS,
    чтобы эффект не терялся при смене платформенного стиля.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        corner_radius: int = 16,
        glow_enabled: bool = True,
        glow_color: QColor | None = None,
    ) -> None:
        super().__init__(parent)
        self._corner_radius = corner_radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)

        if glow_enabled:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(32)
            shadow.setXOffset(0)
            shadow.setYOffset(4)
            shadow.setColor(glow_color or PALETTE.accent_glow)
            self.setGraphicsEffect(shadow)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, self._corner_radius, self._corner_radius)

        # Полупрозрачная заливка с лёгким вертикальным градиентом — имитирует
        # преломление света на стекле, вместо плоского однотонного fill.
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QColor(255, 255, 255, 22))
        gradient.setColorAt(1.0, QColor(255, 255, 255, 10))
        painter.fillPath(path, gradient)

        # Едва заметная светлая окантовка — ключевой признак glassmorphism.
        pen = QPen(PALETTE.border_glass)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawPath(path)

        super().paintEvent(event)
