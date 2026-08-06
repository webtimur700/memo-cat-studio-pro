"""Круглый badge с Viral Score (0-100).

Используется в timeline_view.py (маркер на моменте) и batch_queue_view.py
(колонка результата анализа).
"""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.theme.palette import PALETTE, score_color


class ViralScoreBadge(QWidget):
    def __init__(self, score: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(56, 56)
        self._displayed_score = 0.0
        self._target_score = max(0, min(100, score))

        self._animation = QPropertyAnimation(self, b"displayedScore", self)
        self._animation.setDuration(600)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_score(self, score: int) -> None:
        self._target_score = max(0, min(100, score))
        self._animation.stop()
        self._animation.setStartValue(self._displayed_score)
        self._animation.setEndValue(float(self._target_score))
        self._animation.start()

    def get_displayed_score(self) -> float:
        return self._displayed_score

    def set_displayed_score(self, value: float) -> None:
        self._displayed_score = value
        self.update()

    displayedScore = Property(float, get_displayed_score, set_displayed_score)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(4, 4, self.width() - 8, self.height() - 8)
        color = score_color(int(self._displayed_score))

        # Фоновое кольцо
        bg_pen = QPen(QColor(255, 255, 255, 20))
        bg_pen.setWidth(5)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Прогресс-дуга, начинается сверху (90 * 16 в системе Qt) и идёт по часовой
        progress_pen = QPen(color)
        progress_pen.setWidth(5)
        progress_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(progress_pen)
        span_angle = int(-360 * 16 * (self._displayed_score / 100.0))
        painter.drawArc(rect, 90 * 16, span_angle)

        # Число по центру
        painter.setPen(QPen(PALETTE.text_primary))
        font = QFont()
        font.setBold(True)
        font.setPointSize(13)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(int(self._displayed_score)))
