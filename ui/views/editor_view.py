"""Экран редактора/результатов: список клипов слева, плеер и таймлайн в центре,
заголовки/описание/хештеги справа."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ui.viewmodels.clip_results import ClipResult
from ui.views.preview_player import PreviewPlayer
from ui.views.timeline_view import TimelineMoment, TimelineView
from ui.widgets.clip_details import ClipDetailsWidget
from ui.widgets.clip_list import ClipListWidget


class EditorView(QWidget):
    """Слева — готовые клипы (обложка, Viral Score), по центру — плеер выбранного клипа и
    таймлайн моментов исходного видео, справа — 10 заголовков, описание и хештеги."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.clip_list = ClipListWidget(self)
        self.preview = PreviewPlayer(self)
        self.timeline = TimelineView(self)
        self.details = ClipDetailsWidget(self)
        self._current: ClipResult | None = None

        self.clip_list.clip_selected.connect(self._on_clip_selected)
        self.clip_list.itemClicked.connect(lambda _item: self.preview.play())   # клик пользователя — воспроизвести
        self.timeline.moment_selected.connect(self._on_timeline_moment_selected)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(16)
        center_layout.addWidget(self.preview, stretch=1)
        center_layout.addWidget(self.timeline)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.clip_list)
        splitter.addWidget(center)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 520, 400])
        splitter.setChildrenCollapsible(False)
        self.clip_list.setMinimumWidth(300)
        self.details.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(splitter)

    @property
    def current(self) -> ClipResult | None:
        return self._current

    def add_results(self, results: list[ClipResult], select_first: bool = True) -> None:
        self.clip_list.add_results(results, select_first=select_first)

    def _on_clip_selected(self, result: ClipResult) -> None:
        self._current = result
        self.details.set_clip(result)
        self.preview.load_video(Path(result.clip_path))
        self._refresh_timeline(result)

    def _refresh_timeline(self, result: ClipResult) -> None:
        siblings = [r for r in self.clip_list.results if r.source_video == result.source_video]
        if not siblings:
            return
        self.timeline.set_moments(
            [TimelineMoment(r.start_sec, r.end_sec, r.viral_score, r.title) for r in siblings],
            total_duration_sec=max(r.end_sec for r in siblings),
        )

    def _on_timeline_moment_selected(self, start_sec: float, _end_sec: float) -> None:
        source = self._current.source_video if self._current else None
        self.clip_list.select_by_time(source, start_sec)
