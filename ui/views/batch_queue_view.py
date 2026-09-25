"""Очередь массовой обработки — таблица с реальными прогресс-барами на строку.

Подключается к queue/progress_bus.py (Qt Signals, Шаг 9): каждое обновление
прогресса конкретного видео обновляет соответствующую строку по job_id,
без пересоздания всей таблицы (важно при 100+ видео в очереди).
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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


_FAILED_ROLE = Qt.ItemDataRole.UserRole + 1


class ProgressDelegate(QStyledItemDelegate):
    """Рисует полосу прогресса прямо в ячейке. На каждую строку не создаётся QProgressBar-виджет:
    при 100+ видео тысячи живых виджетов тормозят интерфейс, а делегат рисует только видимые строки."""

    def paint(self, painter, option, index) -> None:  # noqa: N802
        bar = QStyleOptionProgressBar()
        bar.rect = option.rect.adjusted(4, 4, -4, -4)
        bar.minimum, bar.maximum = 0, 100
        bar.progress = int(index.data(Qt.ItemDataRole.UserRole) or 0)
        bar.text = f"{bar.progress}%"
        bar.textVisible = True
        bar.state = option.state
        if index.data(_FAILED_ROLE):
            bar.palette.setColor(bar.palette.ColorRole.Highlight, Qt.GlobalColor.red)
        QApplication.style().drawControl(QStyle.ControlElement.CE_ProgressBar, bar, painter)


class BatchQueueView(QWidget):
    COL_NAME, COL_STAGE, COL_PROGRESS, COL_SCORE, COL_NOTE = range(5)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(["Видео", "Стадия", "Прогресс", "Найдено моментов", "Заметка"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        # ResizeToContents пересчитывает ширину по ВСЕМ строкам при каждом setText: на 1000 видео это ~6 мс на обновление
        header.setSectionResizeMode(self.COL_STAGE, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_SCORE, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_NOTE, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self.COL_NOTE, 260)
        self._table.setColumnWidth(self.COL_STAGE, 200)
        self._table.setColumnWidth(self.COL_SCORE, 150)
        self._table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._table.setColumnWidth(self.COL_PROGRESS, 220)
        self._table.setItemDelegateForColumn(self.COL_PROGRESS, ProgressDelegate(self._table))
        self._table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)

        self._row_by_job_id: dict[str, int] = {}

        # предупреждение про LLM над таблицей: видно сразу, как только выяснилось, что модель не загрузится
        self._banner = QLabel()
        self._banner.setObjectName("llmBanner")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet("color: #f0b429; background: rgba(240,180,41,0.12); border-radius: 10px; padding: 10px;")
        self._banner.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self._banner)
        layout.addWidget(self._table)

    def add_job(self, job_id: str, video_name: str) -> None:
        self.add_jobs([(job_id, video_name)])

    def add_jobs(self, jobs: list[tuple[str, str]]) -> None:
        """Добавляет пачку строк за одну перерисовку (100+ видео за раз не вызывают 100 перерисовок)."""
        self._table.setUpdatesEnabled(False)
        try:
            start = self._table.rowCount()
            self._table.setRowCount(start + len(jobs))
            for offset, (job_id, video_name) in enumerate(jobs):
                row = start + offset
                self._row_by_job_id[job_id] = row
                self._table.setItem(row, self.COL_NAME, QTableWidgetItem(video_name))
                self._table.setItem(row, self.COL_STAGE, QTableWidgetItem(JobStage.QUEUED.value))
                progress_item = QTableWidgetItem()
                progress_item.setData(Qt.ItemDataRole.UserRole, 0)
                self._table.setItem(row, self.COL_PROGRESS, progress_item)
                self._table.setItem(row, self.COL_SCORE, QTableWidgetItem("—"))
                self._table.setItem(row, self.COL_NOTE, QTableWidgetItem(""))
        finally:
            self._table.setUpdatesEnabled(True)

    def set_llm_banner(self, issue) -> None:
        """Общее предупреждение про LLM (None — убрать): причина с цифрами и что можно сделать."""
        if issue is None:
            self._banner.clear()
            self._banner.hide()
            return
        self._banner.setText(f"⚠ {issue.message}\nЧто сделать: {issue.hint}")
        self._banner.show()

    def banner_text(self) -> str:
        return self._banner.text() if not self._banner.isHidden() else ""

    def set_job_warning(self, job_id: str, issue) -> None:
        row = self._row_by_job_id.get(job_id)
        item = self._table.item(row, self.COL_NOTE) if row is not None else None
        if item is not None:
            item.setText("⚠ без LLM: заголовки по умолчанию" if issue.kind != "partial" else "⚠ часть данных от LLM не получена")
            item.setToolTip(f"{issue.message}\nЧто сделать: {issue.hint}")

    def warning_text(self, job_id: str) -> str:
        row = self._row_by_job_id.get(job_id)
        item = self._table.item(row, self.COL_NOTE) if row is not None else None
        return item.text() if item is not None else ""

    def stage_text(self, job_id: str) -> str:
        row = self._row_by_job_id.get(job_id)
        item = self._table.item(row, self.COL_STAGE) if row is not None else None
        return item.text() if item is not None else ""

    def row_count(self) -> int:
        return self._table.rowCount()

    def update_progress(self, job_id: str, stage: JobStage, moments_found: int | None = None) -> None:
        row = self._row_by_job_id.get(job_id)
        if row is None:
            return

        stage_item = self._table.item(row, self.COL_STAGE)
        if stage_item is not None:
            stage_item.setText(stage.value)

        progress_item = self._table.item(row, self.COL_PROGRESS)
        if progress_item is not None:
            stage_index = _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else 0
            progress_item.setData(Qt.ItemDataRole.UserRole, int(stage_index / (len(_STAGE_ORDER) - 1) * 100))
            if stage == JobStage.FAILED:
                progress_item.setData(_FAILED_ROLE, True)

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
            stage_item.setToolTip(error_message)
        self.update_progress(job_id, JobStage.FAILED)
