"""Общий проход декодера (сцены + сканирование): те же сцены и те же кадры, что у раздельного чтения."""

import subprocess
import threading
import time

import numpy as np
import pytest

from video.frame_extractor import FrameExtractor
from video.scene_detector import SceneDetector
from video.shared_decode import SharedPass, scan_indices_supported


def _make_video(path, rate: str, seconds: float = 3.0, cuts: bool = True):
    """Три плана разной картинки (жёсткие склейки) либо один непрерывный план."""
    if cuts:
        graph = (
            f"color=c=red:s=640x360:r={rate}:d={seconds},format=yuv420p[a];"
            f"testsrc2=s=640x360:r={rate}:d={seconds},format=yuv420p[b];"
            f"mandelbrot=s=640x360:r={rate},trim=duration={seconds},format=yuv420p[c];"
            "[a][b][c]concat=n=3:v=1:a=0[out]"
        )
        args = ["-filter_complex", graph, "-map", "[out]"]
    else:
        args = ["-f", "lavfi", "-i", f"testsrc2=s=640x360:r={rate}:d={seconds * 3}"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="module", params=["60", "30000/1001", "25"], ids=["60fps", "29.97fps", "25fps"])
def cut_video(request, tmp_path_factory):
    return _make_video(tmp_path_factory.mktemp("sd") / "cuts.mp4", request.param)


@pytest.fixture(scope="module")
def plain_video(tmp_path_factory):
    return _make_video(tmp_path_factory.mktemp("sd") / "plain.mp4", "30", cuts=False)


def _tuples(segments):
    return [(s.index, s.start_frame, s.end_frame, s.start_sec, s.end_sec) for s in segments]


def _duration(path) -> float:
    with FrameExtractor(path) as extractor:
        return extractor.frame_count / extractor.fps


@pytest.mark.parametrize("scan_fps", [1.0, 2.0])
def test_scan_frames_are_the_same_as_frames_in_range(cut_video, scan_fps):
    duration = _duration(cut_video)
    with FrameExtractor(cut_video) as extractor:
        expected = list(extractor.frames_in_range(0.0, duration, sample_fps=scan_fps))
    with SharedPass(cut_video, scan_fps, duration, scenes=SceneDetector()) as shared:
        actual = list(shared.scan_frames())
        shared.finish_scenes()
    assert len(actual) == len(expected) > 3
    for (t_a, f_a), (t_e, f_e) in zip(actual, expected):
        assert t_a == t_e
        assert np.array_equal(f_a, f_e)


def test_scenes_are_the_same_as_scene_detector_on_the_file(cut_video):
    expected = SceneDetector().detect(cut_video)
    with SharedPass(cut_video, 1.0, _duration(cut_video), scenes=SceneDetector()) as shared:
        list(shared.scan_frames())
        actual = shared.finish_scenes()
    assert len(expected) >= 3                      # склейки действительно найдены — сравнение осмысленное
    assert _tuples(actual) == _tuples(expected)


def test_scenes_of_a_video_without_cuts_match(plain_video):
    expected = SceneDetector().detect(plain_video)
    with SharedPass(plain_video, 1.0, _duration(plain_video), scenes=SceneDetector()) as shared:
        list(shared.scan_frames())
        actual = shared.finish_scenes()
    assert _tuples(actual) == _tuples(expected) and len(actual) == 1


def test_scan_ends_at_end_sec_while_scenes_cover_the_whole_video(cut_video):
    duration = _duration(cut_video)
    with SharedPass(cut_video, 1.0, 4.0, scenes=SceneDetector()) as shared:
        frames = list(shared.scan_frames())
        segments = shared.finish_scenes()
    assert [t for t, _ in frames] == [0.0, 1.0, 2.0, 3.0]
    assert segments[-1].end_sec == pytest.approx(duration, abs=0.1)


def test_without_scene_detector_only_scanning_runs(cut_video):
    with SharedPass(cut_video, 1.0, 4.0, scenes=None) as shared:
        assert len(list(shared.scan_frames())) == 4
        assert shared.finish_scenes() is None


def test_scene_detector_failure_does_not_break_scanning(cut_video):
    class Broken(SceneDetector):
        def stream(self, *args, **kwargs):
            raise RuntimeError("нет scenedetect")

    with SharedPass(cut_video, 1.0, 4.0, scenes=Broken()) as shared:
        assert len(list(shared.scan_frames())) == 4
        assert shared.finish_scenes() is None


def test_scene_worker_error_does_not_hang_the_reader(cut_video, monkeypatch):
    from video import scene_detector

    def boom(self, frame_num, frame):
        if frame_num == 5:
            raise RuntimeError("сбой детектора")

    monkeypatch.setattr(scene_detector.SceneStream, "feed", boom)
    with SharedPass(cut_video, 1.0, 4.0, scenes=SceneDetector()) as shared:
        assert len(list(shared.scan_frames())) == 4
        assert shared.finish_scenes() is None


def test_leaving_early_stops_the_threads(cut_video):
    before = threading.active_count()
    with SharedPass(cut_video, 2.0, 100.0, scenes=SceneDetector()) as shared:
        iterator = shared.scan_frames()
        next(iterator)
    deadline = time.time() + 5
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= before


def test_scan_indices_support_matches_frame_extractor_rule():
    assert scan_indices_supported(60.0, 1.0) and scan_indices_supported(29.97, 2.0)
    assert not scan_indices_supported(30.0, 60.0)        # чаще кадров видео
    assert not scan_indices_supported(30.0, 0.05)        # реже 300 кадров: FrameExtractor перематывает
