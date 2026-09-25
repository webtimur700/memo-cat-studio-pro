"""Фоновый запуск PipelineRunner (Шаги 5-10) из UI, без заморозки интерфейса.

PipelineRunner — синхронный, тяжёлый (декодирование видео, ML-инференс,
ffmpeg-вызовы) — если вызвать его прямо в обработчике сигнала UI-потока,
окно перестанет отвечать на весь пайплайна. QThread с сигналами прогресса —
стандартный Qt-паттерн для такого случая: сигналы, испущенные из рабочего
потока, Qt сам маршрутизирует в UI-поток (queued connection) — безопасно
обновлять виджеты из слотов, подключённых к этим сигналам.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import QThread, Signal

from core.entities.clip import Clip
from core.entities.settings import UserSettings
from pipeline.pipeline_runner import PipelineRunner
from pipeline.shared_models import SharedModels



class PipelineWorker(QThread):
    stage_changed = Signal(str, str, dict)   # job_id, stage_key, data
    job_finished = Signal(str, list)         # job_id, list[Clip]
    job_failed = Signal(str, str)            # job_id, сообщение об ошибке

    def __init__(
        self,
        job_id: str,
        video_path: Path,
        settings: UserSettings,
        models_dir: Path,
        output_dir: Path,
        llm_provider: object | None = None,
        shared_models: SharedModels | None = None,
        plugin_registry=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._job_id = job_id
        self._video_path = video_path
        self._settings = settings
        self._models_dir = models_dir
        self._output_dir = output_dir
        self._llm_provider = llm_provider
        self._shared_models = shared_models
        self._plugin_registry = plugin_registry

    def run(self) -> None:  # выполняется в отдельном потоке — тяжёлая работа здесь безопасна
        def on_progress(stage: str, data: dict) -> None:
            self.stage_changed.emit(self._job_id, stage, data)

        try:
            # LLM (ManagedLMStudio) — общая на все задачи: модель выбирается и загружается один раз.
            # Если LM Studio не запущена, раннер ловит ошибку, пишет warning и берёт заголовок по умолчанию.
            runner = PipelineRunner(
                models_dir=self._models_dir,
                output_dir=self._output_dir,
                llm_provider=self._llm_provider,
                shared_models=self._shared_models,
                plugin_registry=self._plugin_registry,
            )
            clips: list[Clip] = runner.process_video(self._video_path, self._settings, progress=on_progress)
            self.job_finished.emit(self._job_id, clips)
        except Exception as exc:
            logger.error("Пайплайн для {} завершился ошибкой: {}", self._video_path, exc)
            self.job_failed.emit(self._job_id, str(exc))
