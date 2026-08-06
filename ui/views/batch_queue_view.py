"""Очередь массовой обработки — таблица с реальными прогресс-барами на строку.

Подключается к queue/progress_bus.py (Qt Signals, Шаг 9): каждое обновление
прогресса конкретного видео обновляет соответствующую строку по job_id,
без пересоздания всей таблицы (важно при 100+ видео в очереди).
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class JobStage(str, Enum):
    QUEUED = "В очереди"
    ANALYZING = "Анализ"
    SCORING = "Оценка моментов"
    CUTTING = "Нарезка"
    REFRAMING = "Кадрирование"
    SUBTITLING = "Субтитры"
    BRANDING = "Брендирование"
    EXPORTING = "Экспорт"
    DONE = "Готово"
    FAILED = "Ошибка"


_STAGE_ORDER = [
    JobStage.QUEUED,
    JobStage.ANALYZING,
    JobStage.SCORING,
    JobStage.CUTTING,
    JobStage.REFRAMING,
    JobStage.SUBTITLING,
    JobStage.BRANDING,
    JobStage.EXPORTING,
    JobStage.DONE,
]


class BatchQueueView(QWidget):
    COL_NAME, COL_STAGE, COL_PROGRESS, COL_SCORE = range(4)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Видео", "Стадия", "Прогресс", "Найдено моментов"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_STAGE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_SCORE, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setColumnWidth(self.COL_PROGRESS, 220)

        self._row_by_job_id: dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self._table)

    def add_job(self, job_id: str, video_name: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._row_by_job_id[job_id] = row

        self._table.setItem(row, self.COL_NAME, QTableWidgetItem(video_name))
        self._table.setItem(row, self.COL_STAGE, QTableWidgetItem(JobStage.QUEUED.value))

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        self._table.setCellWidget(row, self.COL_PROGRESS, progress_bar)

        self._table.setItem(row, self.COL_SCORE, QTableWidgetItem("—"))

    def update_progress(self, job_id: str, stage: JobStage, moments_found: int | None = None) -> None:
        row = self._row_by_job_id.get(job_id)
        if row is None:
            return

        stage_item = self._table.item(row, self.COL_STAGE)
        if stage_item is not None:
            stage_item.setText(stage.value)

        progress_bar: QProgressBar = self._table.cellWidget(row, self.COL_PROGRESS)  # type: ignore[assignment]
        if progress_bar is not None:
            stage_index = _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else 0
            percent = int(stage_index / (len(_STAGE_ORDER) - 1) * 100)
            progress_bar.setValue(percent)
            if stage == JobStage.FAILED:
                progress_bar.setStyleSheet(
                    "QProgressBar::chunk { background-color: #e65a5a; }"
                )

        if moments_found is not None:
            score_item = self._table.item(row, self.COL_SCORE)
            if score_item is not None:
                score_item.setText(str(moments_found))

    def mark_failed(self, job_id: str, error_message: str) -> None:
        row = self._row_by_job_id.get(job_id)
        if row is None:
            return
        stage_item = self._table.item(row, self.COL_STAGE)
        if stage_item is not None:
            stage_item.setText(f"{JobStage.FAILED.value}: {error_message}")
        self.update_progress(job_id, JobStage.FAILED)
