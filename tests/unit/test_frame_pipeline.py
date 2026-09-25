"""Чтение кадров подряд, фоновый декодер и параллельная детекция: тот же результат, что у простого последовательного кода."""

import subprocess
import threading
import time

import numpy as np
import pytest

from video.frame_extractor import FrameExtractor, prefetched
from vision.parallel_detect import detect_frames, detected_stream


@pytest.fixture(scope="module")
def counter_video(tmp_path_factory):
    """60 fps, 8 с: каждый кадр — свой оттенок серого, по нему видно, какой именно кадр прочитан."""
    path = tmp_path_factory.mktemp("v") / "counter.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=60:duration=8",
                    "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.parametrize("fps", [1.0, 2.0, 5.0, 7.5])
def test_sequential_sampling_returns_the_same_frames_as_seeking(counter_video, fps):
    with FrameExtractor(counter_video) as extractor:
        fast = list(extractor.frames_in_range(1.0, 6.0, sample_fps=fps))
        seek = list(extractor._seek_frames_in_range(1.0, 6.0, 1.0 / fps))
    assert [round(t, 6) for t, _ in fast] == [round(t, 6) for t, _ in seek]
    assert len(fast) == int(np.ceil(5.0 * fps - 1e-9))
    for (_, a), (_, b) in zip(fast, seek):
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 1.0      # тот же кадр (допуск на округление позиции)


def test_sampling_stops_at_the_end_of_the_video(counter_video):
    with FrameExtractor(counter_video) as extractor:
        frames = list(extractor.frames_in_range(6.0, 20.0, sample_fps=2.0))
    assert 3 <= len(frames) <= 5 and all(t < 8.0 for t, _ in frames)


def test_prefetched_keeps_order_and_content(counter_video):
    with FrameExtractor(counter_video) as extractor:
        plain = list(extractor.frames_in_range(0.0, 4.0, sample_fps=3.0))
    with FrameExtractor(counter_video) as extractor:
        ahead = list(prefetched(extractor.frames_in_range(0.0, 4.0, sample_fps=3.0), depth=2))
    assert len(plain) == len(ahead) == 12
    assert all(a[0] == b[0] and (a[1] == b[1]).all() for a, b in zip(plain, ahead))


def test_prefetched_stops_its_thread_when_the_consumer_leaves_early():
    produced = []

    def endless():
        i = 0
        while True:
            produced.append(i)
            yield i
            i += 1

    before = threading.active_count()
    iterator = prefetched(endless(), depth=2)
    assert [next(iterator) for _ in range(3)] == [0, 1, 2]
    iterator.close()
    time.sleep(0.2)
    assert threading.active_count() <= before and len(produced) < 20


def test_prefetched_passes_decoder_errors_to_the_consumer():
    def broken():
        yield 1
        raise RuntimeError("декодер упал")

    iterator = prefetched(broken())
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="декодер упал"):
        next(iterator)


class _SlowDetector:
    """Детекция занимает разное время: порядок результатов всё равно должен совпасть с порядком кадров."""

    def __init__(self):
        self.threads = set()

    def detect(self, frame):
        self.threads.add(threading.get_ident())
        time.sleep(0.03 if int(frame[0, 0, 0]) % 2 else 0.002)
        return [int(frame[0, 0, 0])]


def _frames(n=20):
    return [(i * 0.5, np.full((4, 4, 3), i, np.uint8)) for i in range(n)]


def test_parallel_detection_preserves_order_and_uses_several_threads():
    detector = _SlowDetector()
    result = list(detect_frames(detector, iter(_frames()), workers=4))
    assert [r[2] for r in result] == [[i] for i in range(20)] and [r[0] for r in result] == [i * 0.5 for i in range(20)]
    assert len(detector.threads) > 1


def test_detection_without_detector_and_with_one_worker():
    assert [d for _, _, d in detect_frames(None, iter(_frames(3)))] == [[], [], []]
    assert [d for _, _, d in detect_frames(_SlowDetector(), iter(_frames(3)), workers=1)] == [[0], [1], [2]]


def test_detected_stream_closes_cleanly_on_early_exit(counter_video):
    with FrameExtractor(counter_video) as extractor:
        with detected_stream(_SlowDetectorForVideo(), extractor.frames_in_range(0.0, 8.0, sample_fps=5.0)) as stream:
            first = next(stream)
        assert first[0] == 0.0
        assert len(list(extractor.frames_in_range(0.0, 1.0, sample_fps=2.0))) == 2      # capture ещё жив и никем не занят


class _SlowDetectorForVideo:
    def detect(self, frame):
        time.sleep(0.005)
        return []
