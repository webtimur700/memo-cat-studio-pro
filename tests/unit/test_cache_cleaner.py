import os
import time
from pathlib import Path

import pytest

from pipeline.cache_cleaner import CacheCleaner

DAY = 86400


class _Disk:
    """Диск с управляемым свободным местом: каждое удалённое имитируется вызовом free()."""

    def __init__(self, free):
        self.free = free

    def __call__(self, _path):
        return self.free


def _touch(path: Path, size=10, age_sec=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    t = time.time() - age_sec
    os.utime(path, (t, t))
    return path


@pytest.fixture()
def env(tmp_path):
    out, logs, tmp = tmp_path / "output", tmp_path / "logs", tmp_path / "systmp"
    for d in (out, logs, tmp):
        d.mkdir()
    return out, logs, tmp


def _results(out: Path, stem="vid_moment1", age_sec=10 * DAY):
    return [
        _touch(out / f"{stem}.mp4", 1000, age_sec), _touch(out / f"{stem}_cover.png", 100, age_sec),
        _touch(out / f"{stem}.json", 10, age_sec), _touch(out / f"{stem}.srt", 5, age_sec), _touch(out / f"{stem}.ass", 5, age_sec),
    ]


def _cleaner(env, disk, **kw):
    out, logs, tmp = env
    return CacheCleaner(out, logs, min_free_bytes=1000, temp_root=tmp, disk_free=disk, **kw)


def test_enough_space_deletes_nothing(env):
    out, logs, tmp = env
    stale = _touch(out / "_tmp_audio_a_0.wav", age_sec=5 * DAY)
    report = _cleaner(env, _Disk(10_000)).clean_if_low()
    assert report.removed == [] and stale.exists() and not report.still_low


def test_low_space_removes_stale_temp_and_old_logs_but_never_results_or_active_log(env):
    out, logs, tmp = env
    results = _results(out)
    stale_audio = _touch(out / "_tmp_audio_a_0.wav", age_sec=2 * DAY)
    stale_ass = _touch(out / "_tmp_subs_clip.ass", age_sec=2 * DAY)
    fresh_audio = _touch(out / "_tmp_audio_running_0.wav", age_sec=5)          # принадлежит идущему видео
    stale_dir = tmp / "memo_cat_export_abc"
    _touch(stale_dir / "base.mp4", 500, 2 * DAY)
    os.utime(stale_dir, (time.time() - 2 * DAY,) * 2)
    foreign_dir = tmp / "someone_elses_tmp"
    _touch(foreign_dir / "f", 10, 2 * DAY)
    active = _touch(logs / "app.log", 50, 30 * DAY)
    old_log = _touch(logs / "app.2026-01-01_00-00-00_000000.log", 50, 30 * DAY)
    new_log = _touch(logs / "app.2026-09-23_00-00-00_000000.log", 50, 1 * DAY)

    disk = _Disk(100)
    cleaner = _cleaner(env, disk)
    real_free = cleaner.free_bytes
    report = cleaner.clean_if_low()   # диск «не освобождается» — доходим до конца безопасных шагов

    assert not stale_audio.exists() and not stale_ass.exists() and not stale_dir.exists() and not old_log.exists()
    assert fresh_audio.exists()                                # свежий временный файл не трогаем
    assert foreign_dir.exists()                                # чужое во временной папке — тем более
    assert active.exists()                                     # активный лог не удаляется
    assert not new_log.exists()                                # 3-й шаг: мало места и после старых логов — все ротированные
    assert all(f.exists() for f in results)                    # клипы, обложки, JSON, SRT, ASS — целы
    assert report.still_low and report.freed_bytes > 0


def test_stops_as_soon_as_enough_space(env):
    out, logs, tmp = env
    stale = _touch(out / "_tmp_audio_a_0.wav", 100, 2 * DAY)
    old_log = _touch(logs / "app.2026-01-01_00-00-00_000000.log", 50, 30 * DAY)

    class _Freeing(_Disk):
        def __call__(self, path):
            return 100 if stale.exists() else 5000   # удалили временный — места хватает

    report = _cleaner(env, _Freeing(0)).clean_if_low()
    assert not stale.exists() and old_log.exists()             # до логов дело не дошло
    assert not report.still_low


def test_recent_logs_survive_the_old_logs_step(env):
    out, logs, tmp = env
    recent = _touch(logs / "app.2026-09-23_00-00-00_000000.log", 50, 1 * DAY)
    old = _touch(logs / "app.2026-01-01_00-00-00_000000.log.zip", 50, 30 * DAY)

    class _Freeing(_Disk):
        def __call__(self, path):
            return 5000 if not old.exists() else 100

    _cleaner(env, _Freeing(0)).clean_if_low()
    assert recent.exists() and not old.exists()


def test_results_are_only_offered_oldest_first_and_deleted_only_when_confirmed(env):
    out, _, _ = env
    _results(out, "old", age_sec=30 * DAY)
    _results(out, "mid", age_sec=10 * DAY)
    _results(out, "new", age_sec=1 * DAY)
    cleaner = _cleaner(env, _Disk(0))

    assert [g.stem for g in cleaner.result_groups()] == ["old", "mid", "new"]
    chosen = cleaner.groups_to_free(needed_bytes=1500)
    assert [g.stem for g in chosen] == ["old", "mid"]          # хватает двух самых старых
    assert chosen[0].size_bytes == 1120 and len(chosen[0].files) == 5

    with pytest.raises(PermissionError):
        cleaner.delete_result_groups(chosen, confirmed=False)
    assert len(list(out.iterdir())) == 15                      # без подтверждения ничего не удалено

    freed = cleaner.delete_result_groups(chosen, confirmed=True)
    assert freed == 2240 and sorted(p.name for p in out.iterdir() if p.suffix == ".mp4") == ["new.mp4"]
    assert (out / "new.json").exists() and not (out / "old.json").exists()


def test_confirmed_delete_refuses_files_outside_output_dir(env, tmp_path):
    out, _, _ = env
    outsider = _touch(tmp_path / "precious.mp4", 100)
    from pipeline.cache_cleaner import ResultGroup

    cleaner = _cleaner(env, _Disk(0))
    cleaner.delete_result_groups([ResultGroup("x", (outsider,), 100, 0.0)], confirmed=True)
    assert outsider.exists()


def test_user_video_named_like_temp_is_not_treated_as_temp(env):
    out, _, _ = env
    clip = _touch(out / "_tmp_holiday_moment1.mp4", 100, 30 * DAY)
    _cleaner(env, _Disk(0)).clean_if_low()
    assert clip.exists()
