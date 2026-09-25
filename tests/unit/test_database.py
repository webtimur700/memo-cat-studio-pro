import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from core.entities.clip import Clip
from core.entities.moment import Moment
from core.entities.settings import UserSettings
from database.db import Database
from database.repositories.history_repository import HistoryRepository
from database.repositories.settings_repository import SettingsRepository


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "sub" / "memo_cat.db")   # папки создаются сами
    yield database
    database.close()


def test_schema_is_created_and_versioned_and_reopening_keeps_data(tmp_path):
    path = tmp_path / "m.db"
    first = Database(path)
    assert first.schema_version >= 1
    first.execute("INSERT INTO settings(section, key, value) VALUES ('export', 'fps', '24')")
    first.close()
    again = Database(path)
    assert again.query("SELECT value FROM settings WHERE key = 'fps'")[0]["value"] == "24"
    again.close()


def test_only_changed_fields_are_stored_and_reload_restores_them(db):
    repo = SettingsRepository(db)
    base = UserSettings()
    edited = base.with_field("export", fps=24, bitrate_mbps=8).with_field("shorts", allowed_durations_sec=(15, 30)).with_field(
        "audio", music_offset_db=-18.0, music_enabled=False
    ).with_field("llm", pipeline_reserve_gb=4.5)
    assert repo.save(edited, base) == 6
    rows = {(r["section"], r["key"]) for r in db.query("SELECT section, key FROM settings")}
    assert rows == {("export", "fps"), ("export", "bitrate_mbps"), ("shorts", "allowed_durations_sec"),
                    ("audio", "music_offset_db"), ("audio", "music_enabled"), ("llm", "pipeline_reserve_gb")}

    loaded = repo.load(base)
    assert loaded == edited and loaded.shorts.allowed_durations_sec == (15, 30)         # кортеж, а не список

    # поле вернули к значению по умолчанию — строка исчезает
    assert repo.save(edited.with_field("export", fps=base.export.fps), base) == 5
    assert repo.load(base).export.fps == base.export.fps


def test_new_yaml_defaults_reach_untouched_fields_and_bad_rows_are_skipped(db):
    repo = SettingsRepository(db)
    old_base = UserSettings()
    repo.save(old_base.with_field("export", fps=24), old_base)
    new_base = old_base.with_field("export", bitrate_mbps=9).with_field("batch", max_concurrent_videos=2)   # yaml изменился
    loaded = repo.load(new_base)
    assert loaded.export.fps == 24 and loaded.export.bitrate_mbps == 9 and loaded.batch.max_concurrent_videos == 2

    db.execute("INSERT OR REPLACE INTO settings VALUES ('export', 'fps', '\"не число\"')")
    db.execute("INSERT INTO settings VALUES ('nosuch', 'x', '1')")
    db.execute("INSERT INTO settings VALUES ('export', 'nosuch', '1')")
    db.execute("INSERT INTO settings VALUES ('export', 'codec_video', '{oops')")
    assert repo.load(new_base) == new_base          # ничего не сломалось, всё по умолчанию


def _clip(tmp_path: Path, name: str, start: float, score: int = 70) -> Clip:
    out = tmp_path / f"{name}.mp4"
    out.write_bytes(b"x")
    (tmp_path / f"{name}.json").write_text("{}")
    return Clip(moment=Moment(start, start + 15, score, 0.5), output_path=out, title=f"T {name}", metadata_path=tmp_path / f"{name}.json")


def test_history_projects_runs_clips_and_restart_visibility(db, tmp_path):
    hist = HistoryRepository(db)
    video = tmp_path / "cats.mp4"
    project_id = hist.upsert_project(video)
    assert hist.upsert_project(video) == project_id                       # тот же файл — тот же проект
    run_id = hist.start_run(project_id)
    hist.finish_run(run_id, project_id, "done", [_clip(tmp_path, "cats_moment1", 10.0), _clip(tmp_path, "cats_moment2", 40.0, 90)])

    failed_project = hist.upsert_project(tmp_path / "broken.mp4")
    hist.finish_run(hist.start_run(failed_project), failed_project, "failed", error="нет звуковой дорожки")

    # «перезапуск»: новое соединение с той же базой
    db2 = Database(db.path)
    hist2 = HistoryRepository(db2)
    projects = {p.name: p for p in hist2.projects()}
    assert projects["cats.mp4"].status == "done" and projects["cats.mp4"].clip_count == 2
    assert projects["broken.mp4"].status == "failed" and "звуковой" in projects["broken.mp4"].error
    assert [c.title for c in hist2.clips_for_project(project_id)] == ["T cats_moment1", "T cats_moment2"]
    assert len(hist2.runs_for_project(project_id)) == 1

    (tmp_path / "cats_moment1.mp4").unlink()                              # файл удалили — в счётчик не попадает
    assert {p.name: p.clip_count for p in hist2.projects()}["cats.mp4"] == 1
    db2.close()


def test_running_runs_from_a_crashed_session_become_interrupted(db, tmp_path):
    hist = HistoryRepository(db)
    pid = hist.upsert_project(tmp_path / "v.mp4")
    hist.start_run(pid)
    assert hist.mark_interrupted() == 1
    assert hist.projects()[0].status == "interrupted"


def test_reprocessing_replaces_clip_rows_instead_of_duplicating(db, tmp_path):
    hist = HistoryRepository(db)
    pid = hist.upsert_project(tmp_path / "v.mp4")
    for _ in range(2):
        rid = hist.start_run(pid)
        hist.finish_run(rid, pid, "done", [_clip(tmp_path, "v_moment1", 5.0)])
    assert len(hist.clips_for_project(pid)) == 1 and len(hist.runs_for_project(pid)) == 2
