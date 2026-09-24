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
from llm.lm_studio_provider import LMStudioConfig, LMStudioProvider
from pipeline.pipeline_runner import PipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._job_id = job_id
        self._video_path = video_path
        self._settings = settings
        self._models_dir = models_dir
        self._output_dir = output_dir

    def run(self) -> None:  # выполняется в отдельном потоке — тяжёлая работа здесь безопасна
        def on_progress(stage: str, data: dict) -> None:
            self.stage_changed.emit(self._job_id, stage, data)

        try:
            # URL/таймаут/модель — из .env (LM_STUDIO_*). Если LM Studio не запущена,
            # раннер ловит ошибку, пишет warning и использует заголовок по умолчанию.
            llm_provider = LMStudioProvider(LMStudioConfig.from_env(PROJECT_ROOT / ".env"))
            runner = PipelineRunner(
                models_dir=self._models_dir, output_dir=self._output_dir, llm_provider=llm_provider
            )
            clips: list[Clip] = runner.process_video(self._video_path, self._settings, progress=on_progress)
            self.job_finished.emit(self._job_id, clips)
        except Exception as exc:
            logger.error("Пайплайн для {} завершился ошибкой: {}", self._video_path, exc)
            self.job_failed.emit(self._job_id, str(exc))
