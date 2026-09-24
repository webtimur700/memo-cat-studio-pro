"""Тяжёлые модели, общие для всех видео очереди: YOLO и Whisper грузятся один раз.

PipelineRunner на каждое видео раньше создавал свой YOLO-детектор и Whisper —
в очереди из десятков видео это десятки загрузок одних и тех же весов. Здесь они
создаются лениво при первом обращении и живут, пока очередь не опустеет (close()).
Транскрипция сериализована замком: одна WhisperModel обслуживает по одному запросу,
даже если одновременно обрабатывается несколько видео. Счётчики *_loads нужны, чтобы
проверить (тестом и вручную) «модель загружена ровно один раз».
"""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger


class SharedModels:
    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._lock = threading.Lock()
        self._detector: object | None = None
        self._detector_tried = False
        self._transcribers: dict[tuple[str, str], object] = {}
        self.transcribe_lock = threading.Lock()
        self.detector_loads = 0
        self.transcriber_loads = 0

    def detector(self) -> object | None:
        """YOLO-детектор или None, если модели нет / она не загрузилась (попытка одна)."""
        with self._lock:
            if not self._detector_tried:
                self._detector_tried = True
                model_path = self._models_dir / "yolo11n.onnx"
                if not model_path.exists():
                    logger.warning(
                        "YOLO11 модель не найдена ({}) — скоринг и автокадрирование "
                        "работают в упрощённом режиме (motion-эвристика, без детекции "
                        "животных). Запустите scripts/download_models.py для полного AI-анализа.",
                        model_path,
                    )
                else:
                    try:
                        from vision.yolo_detector import YoloDetector

                        self._detector = YoloDetector(model_path)
                        self.detector_loads += 1
                        logger.info("YOLO11 загружена (общая для очереди)")
                    except Exception as exc:
                        logger.warning("Не удалось загрузить YOLO11 ({}): {} — работаю без детекции", model_path, exc)
            return self._detector

    def transcriber(self, model_size: str, compute_type: str) -> object:
        key = (model_size, compute_type)
        with self._lock:
            if key not in self._transcribers:
                from subtitles.subtitle_service import WhisperTranscriber

                self._transcribers[key] = WhisperTranscriber(model_size=model_size, compute_type=compute_type)
                self.transcriber_loads += 1
                logger.info("Whisper '{}' ({}) создан (общий для очереди)", model_size, compute_type)
            return self._transcribers[key]

    def close(self) -> None:
        """Очередь опустела — отпускаем модели (память нужна пользователю)."""
        with self._lock:
            self._detector = None
            self._detector_tried = False
            self._transcribers.clear()
