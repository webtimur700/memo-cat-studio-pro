"""Главное окно приложения.

Собирает боковую навигацию + QStackedWidget с экранами (Проекты, Редактор,
Очередь, Настройки), подключает тёмную тему и связывает сигналы между
экранами. Добавление видео в "Проекты" запускает pipeline/pipeline_runner.py
в фоновом QThread (ui/pipeline_worker.py) — прогресс приходит обратно через
Qt-сигналы и обновляет вкладку "Очередь" в реальном времени.
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
from ui.pipeline_worker import PipelineWorker
from ui.views.batch_queue_view import BatchQueueView, JobStage
from ui.views.project_view import ProjectView
from ui.views.editor_view import EditorView
from ui.views.settings_view import SettingsView
from ui.views.timeline_view import TimelineMoment
from ui.viewmodels.clip_results import ClipResult, load_results_from_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXPORT_OUTPUT_DIR = PROJECT_ROOT / "export" / "output"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Memo Cat AI Studio Pro")
        self.resize(1280, 800)

        settings_path = PROJECT_ROOT / "config" / "default_settings.yaml"
        self._settings = UserSettings.load_from_yaml(settings_path)
        self._workers: dict[str, PipelineWorker] = {}
        self._job_videos: dict[str, str] = {}   # job_id -> имя исходного видео

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

        # результаты прошлых запусков (json + клипы в папке экспорта) не пропадают после перезапуска
        self.editor_view.add_results(load_results_from_dir(EXPORT_OUTPUT_DIR))

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
        def on_videos_added(paths: list[Path]) -> None:
            for path in paths:
                job_id = f"job_{abs(hash(str(path)))}_{len(self._workers)}"
                self.batch_view.add_job(job_id, path.name)
                self._job_videos[job_id] = path.name
                self.batch_view.update_progress(job_id, JobStage.QUEUED)

                worker = PipelineWorker(
                    job_id=job_id,
                    video_path=path,
                    settings=self._settings,
                    models_dir=MODELS_DIR,
                    output_dir=EXPORT_OUTPUT_DIR,
                    parent=self,
                )
                worker.stage_changed.connect(self._on_pipeline_stage_changed)
                worker.job_finished.connect(self._on_pipeline_finished)
                worker.job_failed.connect(self._on_pipeline_failed)
                # QThread уничтожается Python-сборщиком мусора, если на него
                # не держать ссылку — реальный, часто встречающийся баг в
                # PySide6/PyQt. Держим воркер в self._workers до завершения.
                self._workers[job_id] = worker
                worker.start()

                logger.info("Пайплайн запущен для видео: {}", path)

        self.project_view.videos_added.connect(on_videos_added)

        def on_settings_saved(updated_settings: UserSettings) -> None:
            self._settings = updated_settings
            logger.info(
                "Настройки обновлены: threshold={}, quality={}",
                updated_settings.viral_score.queue_threshold,
                updated_settings.export.quality_preset,
            )

        self.settings_view.settings_saved.connect(on_settings_saved)

    def _on_pipeline_stage_changed(self, job_id: str, stage: str, data: dict) -> None:
        stage_map = {
            "analyzing": JobStage.ANALYZING,
            "scoring": JobStage.SCORING,
            "cutting": JobStage.CUTTING,
            "reframing": JobStage.REFRAMING,
            "done": JobStage.DONE,
        }
        job_stage = stage_map.get(stage)
        if job_stage is None:
            return
        moments_found = data.get("moments_found")
        self.batch_view.update_progress(job_id, job_stage, moments_found=moments_found)

    def _on_pipeline_finished(self, job_id: str, clips: list) -> None:
        self.batch_view.update_progress(job_id, JobStage.DONE, moments_found=len(clips))
        logger.info("Готово: {} клип(ов) для job {}", len(clips), job_id)
        self._workers.pop(job_id, None)
        source_name = self._job_videos.pop(job_id, "")
        if not clips:
            return
        durations = [f"{c.moment.start_sec:.0f}-{c.moment.end_sec:.0f}s (score {c.moment.viral_score})" for c in clips]
        logger.info("Найденные моменты: {}", ", ".join(durations))
        self.editor_view.add_results([ClipResult.from_clip(c, source_name) for c in clips])
        if not self._workers:  # все задачи завершены — показываем результаты
            self.show_page(1)

    def show_page(self, index: int) -> None:
        """Переключает экран и синхронизирует подсветку кнопки в боковой панели."""
        self._stack.setCurrentIndex(index)
        button = self._nav_group.button(index)
        if button is not None:
            button.setChecked(True)

    def _on_pipeline_failed(self, job_id: str, error_message: str) -> None:
        self.batch_view.mark_failed(job_id, error_message)
        self._workers.pop(job_id, None)

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
