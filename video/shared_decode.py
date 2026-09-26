"""Один проход декодера по видео для детекции сцен и сканирования окон.

Раньше видео читалось трижды: PySceneDetect декодировал все кадры (и сам приводил каждый к BGR), сканирование окон снова
декодировало все кадры подряд, чтобы взять один из 60, прыжки/падения — третий раз, окна по 5 с. Теперь первые два делят
один VideoCapture: каждый кадр декодируется и переводится в BGR один раз, уменьшается и уходит в детектор сцен, а каждый
`60/scan_fps`-й ещё и в сканирование (YOLO). Кадры те же, что читали бы по отдельности (тот же OpenCV, те же индексы кадров,
то же округление, те же повторы при сбое `grab`), поэтому сцены, оценки окон и границы моментов не меняются
(тест tests/unit/test_shared_decode.py сверяет побитно).

Прыжки/падения (vision/motion_events.py) сюда не входят: набор окон для них выбирается по результатам сканирования (лучшие 60% по
Viral Score), то есть известен только после прохода. Их вклад в стоимость — в основном YOLO на плотных кадрах, а не декодирование.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
from loguru import logger

from core.entities.scene_segment import SceneSegment
from core.exceptions import VideoDecodeError
from video.frame_extractor import SEEK_GAP_FRAMES
from video.scene_detector import SceneStream

GRAB_RETRIES = 5          # как _max_decode_attempts в PySceneDetect: повторы grab при сбое декодирования
SCENE_QUEUE_DEPTH = 24    # кадров 1080p в очереди детектора сцен (~6 МБ каждый)
SCAN_QUEUE_DEPTH = 4
_DONE = object()


def scan_indices_supported(fps: float, scan_fps: float) -> bool:
    """Общий проход берёт кадры сканирования подряд (grab); если сэмплы реже SEEK_GAP_FRAMES, FrameExtractor перематывает
    (seek) — там выгоднее старый путь. Так же и при сэмплировании чаще кадров видео."""
    frames_per_step = (fps or 30.0) / scan_fps
    return 1.0 <= frames_per_step <= SEEK_GAP_FRAMES


class SharedPass:
    """Читает видео одним потоком и раздаёт кадры двум потребителям.

        with SharedPass(path, scan_fps=1.0, end_sec=duration, scenes=SceneDetector()) as shared:
            ... detected_stream(detector, shared.scan_frames()) ...
            segments = shared.finish_scenes()      # None — детекция сцен не удалась (причина в логе)

    Читатель (`shared-decode`) один; детектор сцен работает в своём потоке (cv2 отпускает GIL). Выход из блока `with` останавливает
    оба потока, даже если потребитель сканирования ушёл раньше времени.
    """

    def __init__(self, path: Path, scan_fps: float, end_sec: float, scenes=None) -> None:
        self._path = path
        self._scan_fps = scan_fps
        self._end_sec = end_sec
        self._scene_detector = scenes
        self._scan_queue: Queue = Queue(maxsize=SCAN_QUEUE_DEPTH)
        self._scene_queue: Queue = Queue(maxsize=SCENE_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._scene_thread: threading.Thread | None = None
        self._reader_error: BaseException | None = None
        self._scene_error: BaseException | None = None
        self._segments: list[SceneSegment] | None = None
        self.frames_read = 0

    # ------------------------------------------------------------------ жизненный цикл
    def __enter__(self) -> "SharedPass":
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise VideoDecodeError(f"OpenCV не смог открыть видео: {self._path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        stream = None
        if self._scene_detector is not None:
            try:
                stream = self._scene_detector.stream(fps or 30.0, size, frame_count)
            except Exception as exc:   # нет scenedetect и т.п.: сканирование идёт без сцен
                logger.warning("Детекция сцен недоступна для {}: {} — считаю 0 смен сцены", self._path.name, exc)
        self._reader = threading.Thread(
            target=self._read, args=(capture, fps or 30.0, frame_count, stream), name="shared-decode", daemon=True
        )
        if stream is not None:
            self._scene_thread = threading.Thread(target=self._detect_scenes, args=(stream,), name="scene-stream", daemon=True)
            self._scene_thread.start()
        self._reader.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        for queue in (self._scan_queue, self._scene_queue):   # освободить производителя, если он ждёт место
            _drain(queue)
        for thread in (self._reader, self._scene_thread):
            if thread is not None:
                thread.join(timeout=30)

    # ------------------------------------------------------------------ потребители
    def scan_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """(время, кадр BGR) для сканирования: как FrameExtractor.frames_in_range(0, end_sec, scan_fps)."""
        while True:
            try:
                item = self._scan_queue.get(timeout=0.1)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            if item is _DONE:
                break
            yield item
        if self._reader_error is not None:
            raise self._reader_error

    def finish_scenes(self) -> list[SceneSegment] | None:
        """Ждёт конца детекции сцен. None — сцены не получились (нет scenedetect / сбой), пайплайн считает 0 смен."""
        if self._scene_thread is None:
            return None
        self._scene_thread.join()
        if self._scene_error is not None:
            logger.warning("Детекция сцен недоступна для {}: {} — считаю 0 смен сцены", self._path.name, self._scene_error)
            return None
        return self._segments

    # ------------------------------------------------------------------ потоки
    def _read(self, capture: cv2.VideoCapture, fps: float, frame_count: int, stream: SceneStream | None) -> None:
        step_sec = 1.0 / self._scan_fps
        frames_per_step = fps * step_sec
        scan_alive = True
        k = 0
        next_scan = 0                 # индекс кадра, который нужен сканированию (round(k * frames_per_step))
        index = 0
        try:
            while not self._stop.is_set():
                grabbed = capture.grab()
                if not grabbed:
                    scan_alive = False   # FrameExtractor.frames_in_range на первом же сбое grab заканчивает сканирование
                    if index < frame_count - 1:   # PySceneDetect повторяет grab, пока не считает, что видео кончилось
                        for _ in range(GRAB_RETRIES):
                            grabbed = capture.grab()
                            if grabbed:
                                break
                    if not grabbed:
                        break
                success, frame = capture.retrieve()
                if not success or frame is None:
                    scan_alive = False
                    index += 1
                    continue
                if stream is not None and not self._put(self._scene_queue, (index, frame)):
                    return
                if scan_alive and index == next_scan:
                    timestamp = k * step_sec
                    if timestamp >= self._end_sec:
                        scan_alive = False
                        self._put(self._scan_queue, _DONE)
                    else:
                        if not self._put(self._scan_queue, (timestamp, frame)):
                            return
                        k += 1
                        next_scan = round(k * frames_per_step)
                    if not scan_alive and stream is None:
                        return         # сканирование закончено и сцены не нужны — дальше читать незачем
                index += 1
            self.frames_read = index
        except BaseException as exc:
            self._reader_error = exc
        finally:
            capture.release()
            self._put(self._scan_queue, _DONE)
            if stream is not None:
                self._put(self._scene_queue, _DONE)

    def _detect_scenes(self, stream: SceneStream) -> None:
        try:
            while True:
                try:
                    item = self._scene_queue.get(timeout=0.1)
                except Empty:
                    if self._stop.is_set():
                        return          # нас остановили (читатель мог уйти, не оставив признака конца)
                    continue
                if item is _DONE:
                    break
                stream.feed(*item)
            self._segments = stream.finish()
        except BaseException as exc:
            self._scene_error = exc
            # читатель не должен зависнуть на полной очереди: вычитываем кадры до конца потока
            while not self._stop.is_set():
                try:
                    if self._scene_queue.get(timeout=0.1) is _DONE:
                        break
                except Empty:
                    continue

    def _put(self, queue: Queue, item) -> bool:
        """put с проверкой остановки; False — нас остановили."""
        while not self._stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False


def _drain(queue: Queue) -> None:
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass
