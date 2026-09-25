"""Список готовых клипов: обложка, Viral Score, заголовок, время в исходном видео."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from ui.theme.palette import score_color
from ui.viewmodels.clip_results import ClipResult

THUMB_W, THUMB_H = 72, 128
ROW_H = THUMB_H + 16


class _ClipRow(QWidget):
    def __init__(self, result: ClipResult) -> None:
        super().__init__()
        thumb = QLabel()
        thumb.setFixedSize(THUMB_W, THUMB_H)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setObjectName("clipThumb")
        pixmap = QPixmap(str(result.cover_path)) if result.cover_path else QPixmap()
        if not pixmap.isNull():
            thumb.setPixmap(
                pixmap.scaled(THUMB_W, THUMB_H, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                              Qt.TransformationMode.SmoothTransformation)
            )
        else:
            thumb.setText("нет\nобложки")

        title = QLabel(result.title or result.clip_path.stem)
        title.setWordWrap(True)
        title.setObjectName("clipTitle")
        meta = QLabel(f"{result.source_video}\n{result.time_range_text}")
        meta.setObjectName("clipMeta")

        color: QColor = score_color(result.viral_score)
        warn = " ⚠ без LLM" if result.llm_issue else ""
        score = QLabel(f"Viral Score {result.viral_score}{warn}")
        if result.llm_issue:
            score.setToolTip(result.llm_issue.text)
        score.setObjectName("clipScore")
        score.setStyleSheet(f"color: {color.name()}; font-weight: 700;")

        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(score)
        text.addWidget(title)
        text.addWidget(meta)
        text.addStretch()

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(10)
        row.addWidget(thumb)
        row.addLayout(text, stretch=1)


class ClipListWidget(QListWidget):
    clip_selected = Signal(object)   # ClipResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[ClipResult] = []
        self.setObjectName("clipList")
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.currentRowChanged.connect(self._on_row_changed)

    @property
    def results(self) -> list[ClipResult]:
        return list(self._results)

    def add_results(self, results: list[ClipResult], select_first: bool = True) -> None:
        """Добавляет результаты (без дублей по пути клипа); новые — сверху."""
        known = {r.clip_path for r in self._results}
        fresh = [r for r in results if r.clip_path not in known]
        if not fresh:
            return
        self.blockSignals(True)
        for result in reversed(fresh):
            self._results.insert(0, result)
            item = QListWidgetItem()
            item.setSizeHint(QSize(300, ROW_H))
            self.insertItem(0, item)
            self.setItemWidget(item, _ClipRow(result))
        self.blockSignals(False)
        if select_first:
            self.setCurrentRow(0)
            self._on_row_changed(0)

    def select_by_time(self, source_video: str | None, start_sec: float) -> None:
        for row, result in enumerate(self._results):
            if abs(result.start_sec - start_sec) < 0.5 and (source_video is None or result.source_video == source_video):
                self.setCurrentRow(row)
                return

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._results):
            self.clip_selected.emit(self._results[row])
