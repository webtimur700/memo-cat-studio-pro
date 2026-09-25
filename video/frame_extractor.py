"""Покадровое извлечение через OpenCV.

Используется vision/yolo_detector.py и effects/cover_generator.py (Шаги 6/8):
анализ не читает весь файл в память, а получает кадры по требованию через
VideoCapture.set(CAP_PROP_POS_MSEC, ...), что на длинных видео (часы записи)
принципиально важно для потребления памяти на 32GB машине при 100+ видео
в очереди.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

from core.exceptions import VideoDecodeError


SEEK_GAP_FRAMES = 300   # реже одного сэмпла на столько кадров — читать подряд дороже, чем перемотать


def prefetched(frames: Iterator, depth: int = 3) -> Iterator:
    """Итератор кадров, который декодируется в фоновом потоке на `depth` кадров вперёд: пока главный поток
    гоняет YOLO по текущему кадру, декодер (cv2 отпускает GIL) готовит следующие. Порядок и содержимое те же.
    Если потребитель бросил итерацию раньше времени, фоновый поток останавливается (generator.close()).
    Вложенный итератор читается ТОЛЬКО из фонового потока, поэтому VideoCapture остаётся однопоточным."""
    queue: Queue = Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def produce() -> None:
        try:
            for item in frames:
                while not stop.is_set():
                    try:
                        queue.put(item, timeout=0.1)
                        break
                    except Full:
                        continue
                if stop.is_set():
                    return
            queue.put(done)
        except BaseException as exc:   # передаём ошибку декодера потребителю
            queue.put(exc)
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()

    thread = threading.Thread(target=produce, daemon=True, name="frame-prefetch")
    thread.start()
    try:
        while True:
            item = queue.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        try:   # освободить производителя, если он ждёт место в очереди
            while True:
                queue.get_nowait()
        except Empty:
            pass
        thread.join(timeout=5)


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

        Позиция задаётся один раз, дальше ненужные кадры пропускаются grab'ом (без конвертации в BGR): пропуск
        кадра 1080p стоит десятки микросекунд, а seek на каждый сэмпл — ~80 мс (декодирование от ключевого кадра).
        Когда между сэмплами больше SEEK_GAP_FRAMES кадров (редкое сэмплирование), выгоднее seek.
        """
        if sample_fps <= 0:
            raise ValueError("sample_fps должен быть положительным")

        step_sec = 1.0 / sample_fps
        source_fps = self.fps or 30.0
        frames_per_step = source_fps * step_sec
        if frames_per_step < 1.0 or frames_per_step > SEEK_GAP_FRAMES:
            yield from self._seek_frames_in_range(start_sec, end_sec, step_sec)
            return

        self._capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        consumed = 0          # сколько кадров от start_sec уже прочитано (grab)
        k = 0
        while True:
            current = start_sec + k * step_sec
            if current >= end_sec:
                break
            target = round(k * frames_per_step)     # индекс кадра относительно start_sec
            while consumed < target:
                if not self._capture.grab():
                    return
                consumed += 1
            if not self._capture.grab():
                return
            consumed += 1
            success, frame = self._capture.retrieve()
            if not success or frame is None:
                return
            yield current, frame
            k += 1

    def _seek_frames_in_range(
        self, start_sec: float, end_sec: float, step_sec: float
    ) -> Iterator[tuple[float, np.ndarray]]:
        current = start_sec
        while current < end_sec:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, current * 1000.0)
            success, frame = self._capture.read()
            if not success or frame is None:
                break
            yield current, frame
            current += step_sec

    def dense_frames_in_range(
        self, start_sec: float, end_sec: float, sample_fps: float
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Кадры диапазона с частотой sample_fps ПОСЛЕДОВАТЕЛЬНЫМ чтением (позиция задаётся один раз, лишние кадры
        пропускаются grab'ом). frames_in_range делает seek на каждый кадр — на 6-8 fps это в разы дороже; нужен
        для анализа движения (прыжки/падения), где между кадрами должно быть 0.1-0.2 с."""
        if sample_fps <= 0:
            raise ValueError("sample_fps должен быть положительным")
        source_fps = self.fps or 30.0
        step = max(1.0, source_fps / sample_fps)
        self._capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        next_index, index = 0.0, 0
        while True:
            timestamp = start_sec + index / source_fps
            if timestamp >= end_sec or not self._capture.grab():
                break
            if index >= next_index:
                success, frame = self._capture.retrieve()
                if success and frame is not None:
                    yield timestamp, frame
                next_index += step
            index += 1

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
