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

    def set_reserve_mib(self, value):
        self.reserve = value

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


def test_low_disk_cleans_temp_silently_but_asks_before_deleting_clips(window, tmp_path, monkeypatch):
    import os

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    stale = out / "_tmp_audio_x_0.wav"
    stale.write_bytes(b"x")
    os.utime(stale, (time.time() - 86400,) * 2)
    clip = out / "keep_moment1.mp4"
    clip.write_bytes(b"clip")
    os.utime(clip, (time.time() - 86400 * 20,) * 2)

    window._cleaner = main_window.CacheCleaner(
        out, tmp_path / "logs", min_free_bytes=10**12, temp_root=tmp_path / "systmp", disk_free=lambda _p: 100
    )
    asked = []
    window._confirm_delete = lambda groups, free: asked.append([g.stem for g in groups]) or False   # пользователь отказывает

    window.project_view.videos_added.emit([Path("/v/a.mp4"), Path("/v/b.mp4")])
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)

    assert not stale.exists()              # временный файл убран без вопросов
    assert clip.exists()                   # клип цел: пользователь отказал
    assert asked == [["keep_moment1"]]     # спросили ровно один раз на всю очередь, а не на каждое видео

    window._cleanup_declined = False
    window._confirm_delete = lambda groups, free: True
    window._ensure_disk_space(queue_idle=True)
    assert not clip.exists()               # подтвердил — удалено


def test_llm_problem_is_shown_in_queue_banner_row_and_clip_card(window, monkeypatch):
    from core.entities.llm_issue import LLMIssue
    from ui.widgets.clip_details import ClipDetailsWidget
    from ui.viewmodels.clip_results import ClipResult

    issue = LLMIssue("no_memory", "LLM не загружена: нужно ~14.4 ГиБ, свободно 13.3 ГиБ.", "Уменьшите запас в настройках.",
                     need_gib=14.4, free_gib=13.3, reserve_gib=6.0)

    class _WarnRunner(_FakeRunner):
        def process_video(self, video_path, settings, progress=None):
            progress("llm_warning", {"issue": issue.to_dict()})
            return super().process_video(video_path, settings, progress)

    monkeypatch.setattr(pipeline_worker, "PipelineRunner", _WarnRunner)
    window.project_view.videos_added.emit([Path("/v/a.mp4")])
    assert _pump(lambda: window.batch_view.warning_text("job_1") != "")
    assert "без LLM" in window.batch_view.warning_text("job_1")
    assert "14.4 ГиБ" in window.batch_view.banner_text() and "Что сделать: Уменьшите запас" in window.batch_view.banner_text()
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)

    # из фонового потока подготовки модели тоже доходит (сигнал), а новая партия баннер сбрасывает
    window._llm_issue_found.emit(None)
    assert _pump(lambda: window.batch_view.banner_text() == "")

    card = ClipDetailsWidget()
    result = ClipResult(clip_path=Path("/v/x.mp4"), source_video="a.mp4", start_sec=0, end_sec=15, viral_score=70,
                        title="Момент 0s (score 70)", llm_issue=issue)
    card.set_clip(result)
    assert not card._llm_note.isHidden() and "свободно 13.3 ГиБ" in card._llm_note.text() and "Что сделать" in card._llm_note.text()
    card.set_clip(ClipResult(clip_path=Path("/v/y.mp4"), source_video="a.mp4", start_sec=0, end_sec=15, viral_score=70,
                             title="Ок", titles=("Ок",)))
    assert card._llm_note.isHidden()


def test_old_clip_json_without_issue_still_explains_missing_titles(tmp_path):
    import json
    from ui.viewmodels.clip_results import ClipResult

    (tmp_path / "old.mp4").write_bytes(b"x")
    (tmp_path / "old.json").write_text(json.dumps({"clip_file": "old.mp4", "title": "Момент 5s (score 60)", "titles": [],
                                                   "llm_errors": ["заголовки: Не удалось подготовить модель"]}), encoding="utf-8")
    result = ClipResult.from_json(tmp_path / "old.json")
    assert result.llm_issue is not None and "Не удалось подготовить модель" in result.llm_issue.message


# --------------------------------------------------------------------------- сохранение настроек и история


def test_settings_survive_restart_and_history_is_visible(window, tmp_path, monkeypatch):
    from core.entities.clip import Clip
    from core.entities.moment import Moment

    # 1. пользователь меняет настройки и сохраняет
    view = window.settings_view
    view._fps_combo.setCurrentText("24")
    view._bitrate_spin.setValue(8)
    view._reserve_spin.setValue(4.5)
    view._music_offset_slider.setValue(-18)
    view._concurrent_spin.setValue(2)
    view._save_button.click()
    assert window._settings.export.fps == 24 and window._scheduler._max_concurrent == 2

    # 2. обрабатывается видео: клип с файлами попадает в историю
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)

    class _ClipRunner(_FakeRunner):
        def process_video(self, video_path, settings, progress=None):
            clip_file = out / f"{Path(video_path).stem}_moment1.mp4"
            clip_file.write_bytes(b"x")
            return [Clip(moment=Moment(10.0, 25.0, 88, 0.5), output_path=clip_file, title="Кот в шоке")]

    monkeypatch.setattr(pipeline_worker, "PipelineRunner", _ClipRunner)
    monkeypatch.setattr(main_window, "EXPORT_OUTPUT_DIR", out)
    window.project_view.videos_added.emit([Path("/v/cats.mp4"), Path("/v/dogs.mp4")])
    assert _pump(lambda: window._scheduler.is_idle and not window._workers)
    assert any("✅" in t and "cats.mp4" in t and "1 клип" in t for t in window.project_view.project_texts())

    # 3. «перезапуск»: новое окно с той же базой
    window._db.close()
    reopened = main_window.MainWindow()
    try:
        assert reopened._settings.export.fps == 24 and reopened._settings.export.bitrate_mbps == 8
        assert reopened._settings.llm.pipeline_reserve_gb == 4.5 and reopened._settings.audio.music_offset_db == -18.0
        assert reopened._settings.batch.max_concurrent_videos == 2
        assert reopened.settings_view._fps_combo.currentText() == "24" and reopened.settings_view._bitrate_spin.value() == 8
        assert reopened.settings_view._reserve_spin.value() == 4.5 and reopened.settings_view._music_offset_slider.value() == -18
        texts = reopened.project_view.project_texts()
        assert any("✅" in t and "cats.mp4" in t and "1 клип" in t for t in texts)      # обработанное видео и число клипов
        assert any("dogs.mp4" in t and "✅" in t for t in texts)
        assert reopened._history.clips_for_project(reopened._history.projects()[0].id) is not None
        reopened._open_project(Path("/v/cats.mp4"))                                     # двойной щелчок: клипы открываются
    finally:
        reopened._db.close()


def test_run_interrupted_by_closing_is_marked_on_next_start(window, monkeypatch):
    import threading

    release = threading.Event()

    class _Hanging(_FakeRunner):
        def process_video(self, video_path, settings, progress=None):
            release.wait(5)
            return []

    monkeypatch.setattr(pipeline_worker, "PipelineRunner", _Hanging)
    window.project_view.videos_added.emit([Path("/v/long.mp4")])
    assert _pump(lambda: window._history.projects() and window._history.projects()[0].status == "running")
    window._db.close()                  # приложение закрыли посреди обработки (без finish_run)
    release.set()
    _pump(lambda: not window._workers, timeout=3)
    reopened = main_window.MainWindow()
    try:
        assert any("⚠️" in t and "long.mp4" in t and "прервано" in t for t in reopened.project_view.project_texts())
    finally:
        reopened._db.close()


def test_example_plugin_is_loaded_at_startup_listed_in_settings_and_selection_survives_restart(window, tmp_path):
    assert "vintage_sepia" in window._plugins.list_effects()
    box = window.settings_view._effect_checkboxes["vintage_sepia"]
    assert box.isEnabled() and not box.isChecked()
    box.setChecked(True)
    window.settings_view._save_button.click()
    assert window._settings.plugins.enabled_effects == ("vintage_sepia",)

    window._db.close()
    reopened = main_window.MainWindow()
    try:
        assert reopened._settings.plugins.enabled_effects == ("vintage_sepia",)
        assert reopened.settings_view._effect_checkboxes["vintage_sepia"].isChecked()
    finally:
        reopened._db.close()


def test_selected_effect_of_a_missing_plugin_is_shown_not_dropped(qapp):
    from core.entities.settings import UserSettings
    from ui.views.settings_view import SettingsView

    settings = UserSettings().with_field("plugins", enabled_effects=("gone_effect",))
    view = SettingsView(settings, available_effects=["vintage_sepia"])
    assert "плагин не найден" in view._effect_checkboxes["gone_effect"].text() and view._effect_checkboxes["gone_effect"].isChecked()
    assert not view._effect_checkboxes["vintage_sepia"].isChecked()
    got = []
    view.settings_saved.connect(got.append)
    view._effect_checkboxes["vintage_sepia"].setChecked(True)
    view._save_button.click()
    assert got[0].plugins.enabled_effects == ("vintage_sepia", "gone_effect") or set(got[0].plugins.enabled_effects) == {"vintage_sepia", "gone_effect"}

    empty = SettingsView(UserSettings())
    assert empty._effect_checkboxes == {}
