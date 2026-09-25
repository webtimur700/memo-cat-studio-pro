"""Экран загрузки видео (Функция 1) и список созданных проектов.

Drag & Drop зона реально принимает файлы (dragEnterEvent/dropEvent), а не
имитирует его текстом — при drop проверяется расширение и эмитится сигнал
videos_dropped(list[Path]) наружу, к viewmodel, которая создаст Project-сущности
через core/interfaces/repository.py (реализация появится на Шаге 6-7).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from database.repositories.history_repository import STATUS_LABELS
from ui.widgets.glass_panel import GlassPanel

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class DropZone(GlassPanel):
    files_dropped = Signal(list)  # list[Path]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, corner_radius=20)
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Перетащите видео сюда")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        subtitle = QLabel("MP4 · MOV · AVI · MKV · WEBM")
        subtitle.setStyleSheet("color: #96969a;")

        self._browse_button = QPushButton("Выбрать файлы…")
        self._browse_button.setObjectName("primaryButton")
        self._browse_button.setFixedWidth(180)
        self._browse_button.clicked.connect(self._on_browse_clicked)

        layout.addStretch()
        layout.addWidget(title, alignment=title.alignment())
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(self._browse_button)
        layout.addStretch()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        valid_paths = [p for p in paths if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if valid_paths:
            self.files_dropped.emit(valid_paths)
        event.acceptProposedAction()

    def _on_browse_clicked(self) -> None:
        filter_str = "Видео (" + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS) + ")"
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Выбрать видео", str(Path.home()), filter_str)
        if file_paths:
            self.files_dropped.emit([Path(p) for p in file_paths])


class ProjectView(QWidget):
    """Основной экран: зона загрузки сверху, список проектов снизу."""

    videos_added = Signal(list)  # list[Path] — наружу, к application-слою
    project_activated = Signal(object)  # Path исходного видео: двойной щелчок — открыть его клипы

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)

        self._drop_zone = DropZone(self)
        self._drop_zone.files_dropped.connect(self._on_files_added)

        list_header = QHBoxLayout()
        list_label = QLabel("Проекты")
        list_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        list_header.addWidget(list_label)
        list_header.addStretch()

        self._project_list = QListWidget()
        self._project_list.setObjectName("surfaceCard")
        self._project_list.setSpacing(4)

        self._project_list.itemActivated.connect(lambda item: self.project_activated.emit(Path(item.data(1000))))
        self._items: dict[str, QListWidgetItem] = {}   # путь -> строка списка (без дублей)

        root_layout.addWidget(self._drop_zone)
        root_layout.addLayout(list_header)
        root_layout.addWidget(self._project_list, stretch=1)

    def _on_files_added(self, paths: list[Path]) -> None:
        for path in paths:
            self.mark_project_status(path, "⏳")
        self.videos_added.emit(paths)

    def load_history(self, records) -> None:
        """Проекты из базы (ProjectRecord): видны после перезапуска, со статусом и числом клипов."""
        for record in reversed(records):   # records: новые сверху; вставляем так, чтобы порядок сохранился
            self.mark_project_status(record.path, STATUS_LABELS.get(record.status, "•"), self._details(record))

    @staticmethod
    def _details(record) -> str:
        from datetime import datetime

        when = datetime.fromtimestamp(record.last_run_at).strftime("%d.%m %H:%M") if record.last_run_at else ""
        parts = []
        if record.clip_count:
            parts.append(f"{record.clip_count} клип(ов)")
        if record.status == "failed" and record.error:
            parts.append(f"ошибка: {record.error[:60]}")
        elif record.status == "interrupted":
            parts.append("прервано при закрытии приложения")
        if when:
            parts.append(when)
        return " · ".join(parts)

    def mark_project_status(self, path: Path, status_label: str, details: str = "") -> None:
        """Строка проекта: «статус имя · подробности»; повторное добавление того же файла обновляет строку."""
        text = f"{status_label}  {path.name}" + (f"  ·  {details}" if details else "")
        item = self._items.get(str(path))
        if item is None:
            item = QListWidgetItem(text)
            item.setData(1000, str(path))
            self._project_list.insertItem(0, item)
            self._items[str(path)] = item
        else:
            item.setText(text)

    def project_texts(self) -> list[str]:
        return [self._project_list.item(i).text() for i in range(self._project_list.count())]
