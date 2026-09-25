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
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.entities.llm_issue import LLMIssue
from core.entities.settings import UserSettings
from llm.lm_studio_provider import LMStudioConfig
from llm.managed_provider import ManagedLMStudio
from pipeline.cache_cleaner import CacheCleaner, ResultGroup
from pipeline.job_scheduler import JobScheduler
from pipeline.shared_models import SharedModels
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
    _llm_issue_found = Signal(object)   # LLMIssue | None: из фонового потока подготовки модели в UI-поток

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Memo Cat AI Studio Pro")
        self.resize(1280, 800)

        settings_path = PROJECT_ROOT / "config" / "default_settings.yaml"
        self._settings = UserSettings.load_from_yaml(settings_path)
        self._workers: dict[str, PipelineWorker] = {}
        self._job_videos: dict[str, str] = {}   # job_id -> имя исходного видео
        self._job_paths: dict[str, Path] = {}   # job_id -> путь к видео, пока задача ждёт очереди или идёт
        self._job_counter = 0
        # YOLO и Whisper общие на всю очередь; LLM тоже одна (ниже). Выгружаются, когда очередь опустела.
        self._shared_models = SharedModels(MODELS_DIR)
        self._cleaner = self._make_cleaner()
        self._cleanup_declined = False   # пользователь отказался удалять клипы — до конца очереди не переспрашиваем
        self._scheduler = JobScheduler(
            self._settings.batch.max_concurrent_videos, self._start_job, on_idle=self._on_queue_idle
        )
        # одна LLM на все задачи: модель выбирается по рейтингу/памяти и загружается один раз (.env: LM_STUDIO_*)
        self._llm = ManagedLMStudio(
            LMStudioConfig.from_env(PROJECT_ROOT / ".env"),
            reserve_mib=self._settings.llm.pipeline_reserve_gb * 1024,
            on_issue=self._llm_issue_found.emit,
        )

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

        self._ensure_disk_space(queue_idle=True)

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
            jobs: list[tuple[str, Path]] = []
            active = set(self._job_paths.values())
            for path in paths:
                if path in active:   # то же видео уже в очереди: его клипы перезаписали бы друг друга
                    logger.info("Пропущено: {} уже в очереди", path.name)
                    continue
                active.add(path)
                self._job_counter += 1
                jobs.append((f"job_{self._job_counter}", path))
            if not jobs:
                return
            if self._scheduler.is_idle:
                self.batch_view.set_llm_banner(None)   # новая партия — причина прошлой больше не актуальна
                self._llm.begin_use()   # пока идёт анализ, модель грузится в фоне; выгрузка — когда очередь опустеет
            self.batch_view.add_jobs([(job_id, path.name) for job_id, path in jobs])
            for job_id, path in jobs:
                self._job_videos[job_id] = path.name
                self._job_paths[job_id] = path
                self._scheduler.submit(job_id)   # свободный слот — запуск сразу, иначе «В очереди»
            logger.info("В очередь добавлено видео: {} (идёт: {}, ждёт: {})", len(paths),
                        self._scheduler.running_count, self._scheduler.waiting_count)

        self.project_view.videos_added.connect(on_videos_added)
        self._llm_issue_found.connect(self.batch_view.set_llm_banner)

        def on_settings_saved(updated_settings: UserSettings) -> None:
            self._settings = updated_settings
            self._scheduler.set_max_concurrent(updated_settings.batch.max_concurrent_videos)
            self._llm.set_reserve_mib(updated_settings.llm.pipeline_reserve_gb * 1024)
            self._cleaner = self._make_cleaner()
            logger.info(
                "Настройки обновлены: threshold={}, quality={}",
                updated_settings.viral_score.queue_threshold,
                updated_settings.export.quality_preset,
            )

        self.settings_view.settings_saved.connect(on_settings_saved)

    def _make_cleaner(self) -> CacheCleaner:
        return CacheCleaner(
            output_dir=EXPORT_OUTPUT_DIR,
            logs_dir=PROJECT_ROOT / "logs",
            min_free_bytes=int(self._settings.cache.min_free_gb * 1e9),
            log_keep_days=self._settings.cache.log_keep_days,
        )

    def _ensure_disk_space(self, queue_idle: bool) -> None:
        """Мало места: удаляем временные файлы и старые логи сами; клипы — только если пользователь подтвердит."""
        if not self._cleaner.is_low_space():
            return
        report = self._cleaner.clean_if_low(temp_age_sec=300 if queue_idle else None)
        if not report.still_low or self._cleanup_declined:
            return
        needed = int(self._settings.cache.min_free_gb * 1e9) - report.free_after
        groups = self._cleaner.groups_to_free(needed)
        if not groups:
            logger.warning("Мало места на диске ({:.1f} ГБ), а удалять нечего", report.free_after / 1e9)
            return
        if self._confirm_delete(groups, report.free_after):
            self._cleaner.delete_result_groups(groups, confirmed=True)
        else:
            self._cleanup_declined = True

    def _confirm_delete(self, groups: list[ResultGroup], free_bytes: int) -> bool:
        total_mb = sum(g.size_bytes for g in groups) / 1e6
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Мало места на диске")
        box.setText(
            f"Свободно {free_bytes / 1e9:.1f} ГБ. Временные файлы и старые логи уже удалены, этого мало.\n"
            f"Удалить {len(groups)} самых старых готовых клипов ({total_mb:.0f} МБ) вместе с обложками, "
            f"JSON и субтитрами? Это необратимо."
        )
        box.setDetailedText("\n".join(g.stem for g in groups))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)   # случайный Enter клипы не удалит
        return box.exec() == QMessageBox.StandardButton.Yes

    def _start_job(self, job_id: str) -> None:
        try:
            self._launch_worker(job_id)
        except Exception as exc:   # не смогли даже запустить поток: пометим ошибкой и пойдём дальше по очереди
            logger.exception("Не удалось запустить обработку {}: {}", job_id, exc)
            self._job_videos.pop(job_id, None)
            self.batch_view.mark_failed(job_id, str(exc))
            QTimer.singleShot(0, lambda: self._finish_job(job_id))

    def _launch_worker(self, job_id: str) -> None:
        self._ensure_disk_space(queue_idle=False)
        path = self._job_paths[job_id]
        worker = PipelineWorker(
            job_id=job_id,
            video_path=path,
            settings=self._settings,
            models_dir=MODELS_DIR,
            output_dir=EXPORT_OUTPUT_DIR,
            llm_provider=self._llm,
            shared_models=self._shared_models,
            parent=self,
        )
        worker.stage_changed.connect(self._on_pipeline_stage_changed)
        worker.job_finished.connect(self._on_pipeline_finished)
        worker.job_failed.connect(self._on_pipeline_failed)
        worker.finished.connect(worker.deleteLater)   # уничтожаем поток только когда он реально остановился
        # QThread уничтожается Python-сборщиком мусора, если на него не держать ссылку — держим до завершения.
        self._workers[job_id] = worker
        self.batch_view.update_progress(job_id, JobStage.ANALYZING)
        worker.start()
        logger.info("Пайплайн запущен для видео: {}", path)

    def _finish_job(self, job_id: str) -> None:
        self._workers.pop(job_id, None)
        self._job_paths.pop(job_id, None)
        self._scheduler.job_done(job_id)   # запускает следующее видео; при пустой очереди зовёт _on_queue_idle

    def _on_queue_idle(self) -> None:
        """Очередь пуста: выгружаем LLM (в фоне — выгрузка не блокирует UI) и общие YOLO/Whisper."""
        import threading

        self._cleanup_declined = False
        self._shared_models.close()
        if self._llm.end_use():
            threading.Thread(target=self._llm.release_if_unused, name="lm-studio-release", daemon=True).start()

    def _on_pipeline_stage_changed(self, job_id: str, stage: str, data: dict) -> None:
        if stage == "llm_warning":
            issue = LLMIssue.from_dict(data.get("issue"))
            if issue is not None:
                self.batch_view.set_job_warning(job_id, issue)
                self.batch_view.set_llm_banner(issue)
            return
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
        source_name = self._job_videos.pop(job_id, "")
        self._finish_job(job_id)
        if not clips:
            return
        durations = [f"{c.moment.start_sec:.0f}-{c.moment.end_sec:.0f}s (score {c.moment.viral_score})" for c in clips]
        logger.info("Найденные моменты: {}", ", ".join(durations))
        # пользователь может смотреть другой клип — новые результаты очереди не перехватывают выбор
        self.editor_view.add_results(
            [ClipResult.from_clip(c, source_name) for c in clips], select_first=self.editor_view.current is None
        )
        if self._scheduler.is_idle:  # вся очередь обработана — показываем результаты
            self.show_page(1)

    def show_page(self, index: int) -> None:
        """Переключает экран и синхронизирует подсветку кнопки в боковой панели."""
        self._stack.setCurrentIndex(index)
        button = self._nav_group.button(index)
        if button is not None:
            button.setChecked(True)

    def _on_pipeline_failed(self, job_id: str, error_message: str) -> None:
        self.batch_view.mark_failed(job_id, error_message)
        self._job_videos.pop(job_id, None)
        self._finish_job(job_id)

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
