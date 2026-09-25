"""Детекция YOLO по потоку кадров в нескольких потоках с сохранением порядка.

Одиночный вызов onnxruntime на 14 потоках почти не масштабируется (~28 мс на кадр 640x640 у yolo11n на Ryzen 7 8845HS),
а несколько одновременных `run` на одной сессии дают ~18 мс на кадр (замер: docs/performance.md): вызов `session.run`
отпускает GIL, сессия потокобезопасна. Результат детекции — чистая функция кадра, поэтому итог тот же, что
при последовательном вызове; порядок кадров сохраняется, число кадров «в полёте» ограничено (память).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from core.entities.detection import Detection
from video.frame_extractor import prefetched

DETECT_WORKERS = 4


def detect_frames(
    detector, frames: Iterable[tuple[float, np.ndarray]], workers: int = DETECT_WORKERS
) -> Iterator[tuple[float, np.ndarray, list[Detection]]]:
    """(timestamp, frame) -> (timestamp, frame, detections) в том же порядке. detector=None — детекций нет."""
    if detector is None:
        for timestamp, frame in frames:
            yield timestamp, frame, []
        return
    if workers <= 1:
        for timestamp, frame in frames:
            yield timestamp, frame, detector.detect(frame)
        return

    pending: deque[tuple[float, np.ndarray, Future]] = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yolo") as pool:
        try:
            for timestamp, frame in frames:
                pending.append((timestamp, frame, pool.submit(detector.detect, frame)))
                if len(pending) >= workers * 2:
                    ts, fr, future = pending.popleft()
                    yield ts, fr, future.result()
            while pending:
                ts, fr, future = pending.popleft()
                yield ts, fr, future.result()
        finally:
            for _, _, future in pending:   # потребитель ушёл раньше времени или ошибка
                future.cancel()


@contextmanager
def detected_stream(
    detector, frames: Iterable[tuple[float, np.ndarray]], workers: int = DETECT_WORKERS, prefetch_depth: int = 8
) -> Iterator[Iterator[tuple[float, np.ndarray, list[Detection]]]]:
    """Поток (timestamp, frame, detections): декодирование в фоне (prefetched) + детекция в потоках.
    При выходе из блока оба конвейера останавливаются, до этого VideoCapture освобождать нельзя."""
    source = prefetched(iter(frames), depth=prefetch_depth)
    stream = detect_frames(detector, source, workers)
    try:
        yield stream
    finally:
        stream.close()
        source.close()
