"""Покадровое извлечение через OpenCV.

Используется vision/yolo_detector.py и effects/cover_generator.py (Шаги 6/8):
анализ не читает весь файл в память, а получает кадры по требованию через
VideoCapture.set(CAP_PROP_POS_MSEC, ...), что на длинных видео (часы записи)
принципиально важно для потребления памяти на 32GB машине при 100+ видео
в очереди.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.exceptions import VideoDecodeError


class FrameExtractor:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise VideoDecodeError(f"OpenCV не смог открыть видео: {path}")

        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __enter__(self) -> "FrameExtractor":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()

    def release(self) -> None:
        if self._capture.isOpened():
            self._capture.release()

    def frame_at(self, timestamp_sec: float) -> np.ndarray:
        """Возвращает один кадр (BGR, как в OpenCV) на указанной секунде."""
        self._capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        success, frame = self._capture.read()
        if not success or frame is None:
            raise VideoDecodeError(
                f"Не удалось прочитать кадр на {timestamp_sec:.2f}s из {self._path}"
            )
        return frame

    def frames_in_range(
        self, start_sec: float, end_sec: float, sample_fps: float = 5.0
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Генератор кадров в диапазоне [start_sec, end_sec) с заданной частотой
        сэмплирования (Функция 2: анализ на 5 fps достаточен для детекции
        движения/объектов и в разы дешевле, чем разбор каждого кадра оригинала).
        """
        if sample_fps <= 0:
            raise ValueError("sample_fps должен быть положительным")

        step_sec = 1.0 / sample_fps
        current = start_sec
        while current < end_sec:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, current * 1000.0)
            success, frame = self._capture.read()
            if not success or frame is None:
                break
            yield current, frame
            current += step_sec

    def best_frame_for_cover(self, start_sec: float, end_sec: float) -> tuple[float, np.ndarray]:
        """Эвристика выбора "лучшего" кадра для обложки (Функция 11):
        берёт кадр с максимальной резкостью (variance of Laplacian) среди
        сэмплированных — надёжный, объективный критерий "не смазанный кадр",
        а не гадание. Выбор по смысловому содержанию (крупная морда животного)
        добавляется поверх этого в effects/cover_generator.py через YOLO-детекцию.
        """
        best_timestamp = start_sec
        best_frame: np.ndarray | None = None
        best_sharpness = -1.0

        for timestamp, frame in self.frames_in_range(start_sec, end_sec, sample_fps=2.0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame
                best_timestamp = timestamp

        if best_frame is None:
            best_frame = self.frame_at(start_sec)

        return best_timestamp, best_frame
