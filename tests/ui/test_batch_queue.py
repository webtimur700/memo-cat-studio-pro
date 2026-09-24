"""Очередь в интерфейсе: по одному видео за раз, остальные «В очереди», общие модели, LLM выгружается в конце."""

import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

import ui.main_window as main_window
import ui.pipeline_worker as pipeline_worker
from ui.views.batch_queue_view import BatchQueueView

WORK_SEC = 0.15


class _FakeLLM:
    def __init__(self, *_a, **_k):
        self.begin = self.end = self.released = 0

    def begin_use(self):
        self.begin += 1

    def end_use(self):
        self.end += 1
        return True

    def release_if_unused(self):
        self.released += 1


class _Tracker:
    running = 0
    peak = 0
    order: list[str] = []
    shared_seen: set[int] = set()


class _FakeRunner:
    def __init__(self, **kwargs):
        self._shared = kwargs["shared_models"]

    def process_video(self, video_path, settings, progress=None):
        _Tracker.running += 1
        _Tracker.peak = max(_Tracker.peak, _Tracker.running)
        _Tracker.order.append(Path(video_path).name)
        _Tracker.shared_seen.add(id(self._shared))
        time.sleep(WORK_SEC)
        _Tracker.running -= 1
        return []


def _pump(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture()
def window(qapp, monkeypatch, tmp_path):
    _Tracker.running = _Tracker.peak = 0
    _Tracker.order = []
    _Tracker.shared_seen = set()
    monkeypatch.setattr(main_window, "ManagedLMStudio", _FakeLLM)
    monkeypatch.setattr(main_window, "EXPORT_OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(pipeline_worker, "PipelineRunner", _FakeRunner)
    win = main_window.MainWindow()
    yield win
    win.close()


def test_three_videos_run_one_at_a_time_others_are_queued(window):
    paths = [Path(f"/v/clip{i}.mp4") for i in range(3)]
    window.project_view.videos_added.emit(paths)

    ids = [f"job_{i}" for i in (1, 2, 3)]
    assert [window.batch_view.stage_text(j) for j in ids] == ["Анализ", "В очереди", "В очереди"]

    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert _Tracker.peak == 1
    assert _Tracker.order == ["clip0.mp4", "clip1.mp4", "clip2.mp4"]
    assert len(_Tracker.shared_seen) == 1   # один SharedModels на всю очередь
    assert window._llm.begin == 1 and window._llm.end == 1
    assert _pump(lambda: window._llm.released == 1)   # выгрузка — один раз, в конце


def test_concurrency_setting_is_honoured(window):
    window._scheduler.set_max_concurrent(2)
    window.project_view.videos_added.emit([Path(f"/v/c{i}.mp4") for i in range(4)])
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert _Tracker.peak == 2


def test_second_batch_while_first_runs_does_not_release_llm_midway(window):
    window.project_view.videos_added.emit([Path("/v/a.mp4"), Path("/v/b.mp4")])
    window.project_view.videos_added.emit([Path("/v/c.mp4")])
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert window._llm.begin == 1 and window._llm.end == 1


def test_same_video_twice_is_queued_once(window):
    window.project_view.videos_added.emit([Path("/v/same.mp4"), Path("/v/same.mp4")])
    window.project_view.videos_added.emit([Path("/v/same.mp4")])   # и в отдельной пачке, пока идёт
    assert window.batch_view.row_count() == 1
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert _Tracker.order == ["same.mp4"]


def test_job_that_cannot_start_is_marked_failed_and_queue_moves_on(window, monkeypatch):
    real = main_window.PipelineWorker
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs["job_id"])
        if len(calls) == 1:
            raise RuntimeError("нет потока")
        return real(**kwargs)

    monkeypatch.setattr(main_window, "PipelineWorker", flaky)
    window.project_view.videos_added.emit([Path("/v/a.mp4"), Path("/v/b.mp4")])
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert window.batch_view.stage_text("job_1").startswith("Ошибка")
    assert _Tracker.order == ["b.mp4"]
    assert window._llm.begin == 1 and window._llm.end == 1


def test_long_list_does_not_freeze_the_ui(qapp):
    view = BatchQueueView()
    view.show()
    jobs = [(f"job_{i}", f"video_{i}.mp4") for i in range(1000)]
    t0 = time.perf_counter()
    view.add_jobs(jobs)
    QCoreApplication.processEvents()
    add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(1000):
        view.update_progress(f"job_{i}", view_stage(i), moments_found=i % 5)
    QCoreApplication.processEvents()
    update_time = time.perf_counter() - t0

    assert view.row_count() == 1000
    assert add_time < 1.0, add_time
    assert update_time < 1.0, update_time


def view_stage(i):
    from ui.views.batch_queue_view import JobStage

    return [JobStage.QUEUED, JobStage.ANALYZING, JobStage.CUTTING, JobStage.DONE][i % 4]
