"""Таймлайн видео с найденными "моментами" (Функция 2/3).

Каждый момент — прямоугольник на шкале времени, цвет которого соответствует
Viral Score (см. ui/theme/palette.score_color). Клик по моменту эмитит сигнал
с его временными границами — preview_player.py на это подписывается и
перематывает предпросмотр.

Реализовано на QGraphicsView/QGraphicsScene — это настоящий интерактивный
таймлайн (zoom колесом мыши, клики, tooltip с точным score), а не статичная
картинка.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from ui.theme.palette import PALETTE, score_color

TIMELINE_HEIGHT = 64
PIXELS_PER_SECOND_DEFAULT = 12.0


@dataclass(slots=True)
class TimelineMoment:
    start_sec: float
    end_sec: float
    viral_score: int
    label: str = ""


class _MomentItem(QGraphicsRectItem):
    def __init__(self, moment: TimelineMoment, pixels_per_second: float) -> None:
        x = moment.start_sec * pixels_per_second
        width = max(2.0, (moment.end_sec - moment.start_sec) * pixels_per_second)
        super().__init__(QRectF(x, 6, width, TIMELINE_HEIGHT - 12))
        self.moment = moment

        color = score_color(moment.viral_score)
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
        self.setPen(QPen(color.lighter(130), 1))
        self.setAcceptHoverEvents(True)
        self.setToolTip(
            f"{moment.label or 'Момент'} · Viral Score: {moment.viral_score}\n"
            f"{moment.start_sec:.1f}s → {moment.end_sec:.1f}s"
        )

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setPen(QPen(QColor(255, 255, 255), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        color = score_color(self.moment.viral_score)
        self.setPen(QPen(color.lighter(130), 1))
        super().hoverLeaveEvent(event)


class TimelineView(QWidget):
    moment_selected = Signal(float, float)  # start_sec, end_sec

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixels_per_second = PIXELS_PER_SECOND_DEFAULT
        self._moments: list[TimelineMoment] = []
        self._auto_fit = True   # масштаб подгоняется под ширину, пока пользователь не менял его колесом
        self._items: list[_MomentItem] = []

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setBackgroundBrush(QBrush(PALETTE.surface))
        self._view.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)   # без центрирования узкой сцены
        self._view.setFixedHeight(TIMELINE_HEIGHT + 16)
        self._view.viewport().installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._scene.selectionChanged.connect(self._emit_selected_moment)   # один раз, а не при каждом set_moments

    def _fit_pixels_per_second(self, total_duration_sec: float) -> float:
        available = max(self._view.viewport().width(), 300) - 4
        return min(200.0, max(2.0, available / max(total_duration_sec, 1.0)))

    def _relayout(self, fit: bool) -> None:
        total_duration = max((m.end_sec for m in self._moments), default=60.0)
        self._render(self._moments, total_duration, fit)

    def set_moments(self, moments: list[TimelineMoment], total_duration_sec: float, fit: bool = True) -> None:
        self._moments = moments
        if fit:
            self._auto_fit = True
        self._render(moments, total_duration_sec, fit)

    def select_moment(self, start_sec: float) -> None:
        """Подсвечивает момент, начинающийся в start_sec (без повторного сигнала) и показывает его."""
        self._scene.blockSignals(True)
        self._scene.clearSelection()
        for item in self._items:
            if abs(item.moment.start_sec - start_sec) < 0.5:
                item.setSelected(True)
                self._view.ensureVisible(item, 40, 0)
        self._scene.blockSignals(False)

    def _render(self, moments: list[TimelineMoment], total_duration_sec: float, fit: bool) -> None:
        if fit and self._auto_fit:
            self._pixels_per_second = self._fit_pixels_per_second(total_duration_sec)
        selected_starts = [it.moment.start_sec for it in self._scene.selectedItems() if isinstance(it, _MomentItem)]
        self._scene.blockSignals(True)
        self._scene.clear()
        self._items = []

        total_width = max(total_duration_sec * self._pixels_per_second, 400.0)
        self._scene.setSceneRect(0, 0, total_width, TIMELINE_HEIGHT)

        track_bg = self._scene.addRect(
            QRectF(0, 0, total_width, TIMELINE_HEIGHT),
            QPen(Qt.PenStyle.NoPen),
            QBrush(PALETTE.background),
        )
        track_bg.setZValue(-1)

        for moment in moments:
            item = _MomentItem(moment, self._pixels_per_second)
            item.setZValue(1)
            item.setFlag(item.GraphicsItemFlag.ItemIsSelectable, True)
            self._scene.addItem(item)
            self._items.append(item)
            if moment.start_sec in selected_starts:
                item.setSelected(True)
        self._scene.blockSignals(False)

    def _emit_selected_moment(self) -> None:
        selected = [it for it in self._scene.selectedItems() if isinstance(it, _MomentItem)]
        if selected:
            m = selected[0].moment
            self.moment_selected.emit(m.start_sec, m.end_sec)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        # ширина viewport меняется уже ПОСЛЕ resize родителя, поэтому следим за самим viewport
        if event.type() == QEvent.Type.Resize and watched is self._view.viewport():
            if self._auto_fit and self._moments:
                self._relayout(fit=True)
            return False
        if event.type() == QWheelEvent.Type.Wheel and watched is self._view.viewport():
            wheel_event: QWheelEvent = event
            self._auto_fit = False   # пользователь взял масштаб в свои руки
            if wheel_event.angleDelta().y() > 0:
                self._pixels_per_second = min(self._pixels_per_second * 1.15, 200.0)
            else:
                self._pixels_per_second = max(self._pixels_per_second / 1.15, 2.0)
            if self._moments:
                self._relayout(fit=False)
            return True
        return False
