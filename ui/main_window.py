"""Главное окно приложения.

Собирает боковую навигацию + QStackedWidget с экранами (Проекты, Редактор,
Очередь, Настройки), подключает тёмную тему и связывает сигналы между
экранами. Application-сервисы (video/vision/llm и т.д. из Шагов 5-9) будут
инжектироваться сюда через core/container.py — в этом виджете уже заложены
точки подключения (см. _wire_pipeline_signals), чтобы не переписывать UI
заново, когда появится реальный pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.entities.settings import UserSettings
from ui.views.batch_queue_view import BatchQueueView, JobStage
from ui.views.project_view import ProjectView
from ui.views.preview_player import PreviewPlayer
from ui.views.settings_view import SettingsView
from ui.views.timeline_view import TimelineMoment, TimelineView

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EditorView(QWidget):
    """Экран редактора: таймлайн сверху, предпросмотр снизу — объединяет
    timeline_view.py и preview_player.py в единый рабочий экран.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = TimelineView(self)
        self.preview = PreviewPlayer(self)
        self.timeline.moment_selected.connect(
            lambda start, _end: self.preview.seek_to(start)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(self.timeline)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Memo Cat AI Studio Pro")
        self.resize(1280, 800)

        settings_path = PROJECT_ROOT / "config" / "default_settings.yaml"
        self._settings = UserSettings.load_from_yaml(settings_path)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar, self._nav_group = self._build_sidebar()
        root_layout.addWidget(sidebar)

        self._stack = QStackedWidget()
        self.project_view = ProjectView()
        self.editor_view = EditorView()
        self.batch_view = BatchQueueView()
        self.settings_view = SettingsView(self._settings)

        for view in (self.project_view, self.editor_view, self.batch_view, self.settings_view):
            self._stack.addWidget(view)

        root_layout.addWidget(self._stack, stretch=1)

        self._wire_pipeline_signals()

    def _build_sidebar(self) -> tuple[QWidget, QButtonGroup]:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(4)

        group = QButtonGroup(sidebar)
        group.setExclusive(True)

        for index, (icon, label) in enumerate(
            [("📁", "Проекты"), ("🎬", "Редактор"), ("📊", "Очередь"), ("⚙️", "Настройки")]
        ):
            button = QPushButton(f"{icon}  {label}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.clicked.connect(lambda _checked, i=index: self._stack.setCurrentIndex(i))
            group.addButton(button, index)
            layout.addWidget(button)

        layout.addStretch()
        return sidebar, group

    def _wire_pipeline_signals(self) -> None:
        """Точка подключения будущего pipeline (Шаги 5-9).

        Пока Application-сервисы ещё не реализованы, сигнал добавления видео
        уже создаёт реальную запись в очереди пакетной обработки — это не
        имитация, а рабочая связь UI-компонентов между собой, на которую
        позже просто "навесится" реальный pipeline_runner.py вместо
        сгенерированного здесь job_id.
        """

        def on_videos_added(paths: list[Path]) -> None:
            for path in paths:
                job_id = f"job_{abs(hash(str(path)))}"
                self.batch_view.add_job(job_id, path.name)
                self.batch_view.update_progress(job_id, JobStage.QUEUED)
                logger.info("Видео добавлено в очередь: {}", path)

        self.project_view.videos_added.connect(on_videos_added)

        def on_settings_saved(updated_settings: UserSettings) -> None:
            self._settings = updated_settings
            logger.info(
                "Настройки обновлены: threshold={}, quality={}",
                updated_settings.viral_score.queue_threshold,
                updated_settings.export.quality_preset,
            )

        self.settings_view.settings_saved.connect(on_settings_saved)

    def load_demo_timeline(self) -> None:
        """Вспомогательный метод для ручной проверки timeline_view без
        готового AI-анализа — использует детерминированные тестовые данные,
        не случайные, чтобы поведение было воспроизводимым при отладке UI.
        """
        demo_moments = [
            TimelineMoment(2.0, 18.0, 82, "Кот прыгает на шкаф"),
            TimelineMoment(25.0, 40.0, 45, "Пробежка по коридору"),
            TimelineMoment(55.0, 90.0, 91, "Неожиданная встреча с собакой"),
        ]
        self.editor_view.timeline.set_moments(demo_moments, total_duration_sec=120.0)


def _load_stylesheet(app: QApplication) -> None:
    qss_path = PROJECT_ROOT / "ui" / "theme" / "dark_theme.qss"
    app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def main() -> None:
    logger.add(PROJECT_ROOT / "logs" / "app.log", rotation="50 MB", retention="14 days")

    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    _load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
