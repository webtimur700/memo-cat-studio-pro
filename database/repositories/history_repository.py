"""Проекты (исходные видео) и история их обработки: запуски и готовые клипы. Видны после перезапуска приложения."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.entities.clip import Clip
from database.db import Database

STATUS_LABELS = {"queued": "⏳", "running": "⚙️", "done": "✅", "failed": "❌", "interrupted": "⚠️"}


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    id: int
    path: Path
    name: str
    added_at: float
    status: str
    error: str | None
    last_run_at: float | None
    clip_count: int


@dataclass(frozen=True, slots=True)
class ClipRecord:
    clip_path: Path
    json_path: Path | None
    title: str
    viral_score: int
    start_sec: float
    end_sec: float


class HistoryRepository:
    def __init__(self, db: Database, now=time.time) -> None:
        self._db = db
        self._now = now

    def mark_interrupted(self) -> int:
        """Запуски, оставшиеся «идущими» с прошлого сеанса (приложение закрыли или оно упало), — прерванные."""
        cursor = self._db.execute("UPDATE runs SET status = 'interrupted', finished_at = ? WHERE status IN ('running', 'queued')", (self._now(),))
        self._db.execute("UPDATE projects SET last_status = 'interrupted' WHERE last_status IN ('running', 'queued')")
        return cursor.rowcount

    def upsert_project(self, path: Path) -> int:
        path = Path(path)
        existing = self._db.query("SELECT id FROM projects WHERE path = ?", (str(path),))
        if existing:
            self._db.execute("UPDATE projects SET last_status = 'queued', last_error = NULL WHERE id = ?", (existing[0]["id"],))
            return existing[0]["id"]
        cursor = self._db.execute("INSERT INTO projects(path, name, added_at) VALUES (?, ?, ?)", (str(path), path.name, self._now()))
        return int(cursor.lastrowid)

    def start_run(self, project_id: int) -> int:
        cursor = self._db.execute("INSERT INTO runs(project_id, started_at, status) VALUES (?, ?, 'running')", (project_id, self._now()))
        self._db.execute("UPDATE projects SET last_status = 'running', last_run_at = ? WHERE id = ?", (self._now(), project_id))
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, project_id: int, status: str, clips: Iterable[Clip] = (), error: str | None = None) -> None:
        clips = list(clips)
        now = self._now()
        self._db.execute("UPDATE runs SET status = ?, finished_at = ?, moments = ?, error = ? WHERE id = ?", (status, now, len(clips), error, run_id))
        self._db.execute("UPDATE projects SET last_status = ?, last_error = ?, last_run_at = ? WHERE id = ?", (status, error, now, project_id))
        for clip in clips:
            # клип с тем же файлом (повторная обработка) заменяет прежнюю запись
            self._db.execute("DELETE FROM clips WHERE clip_path = ?", (str(clip.output_path),))
            self._db.execute(
                "INSERT INTO clips(run_id, project_id, clip_path, json_path, title, viral_score, start_sec, end_sec, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, project_id, str(clip.output_path), str(clip.metadata_path) if clip.metadata_path else None, clip.title,
                 clip.moment.viral_score, clip.moment.start_sec, clip.moment.end_sec, now),
            )

    def projects(self) -> list[ProjectRecord]:
        """Все проекты, новые сверху; clip_count — клипы, файлы которых ещё существуют."""
        rows = self._db.query("SELECT * FROM projects ORDER BY COALESCE(last_run_at, added_at) DESC, id DESC")
        result = []
        for row in rows:
            count = sum(1 for c in self._db.query("SELECT clip_path FROM clips WHERE project_id = ?", (row["id"],)) if Path(c["clip_path"]).is_file())
            result.append(ProjectRecord(row["id"], Path(row["path"]), row["name"], row["added_at"], row["last_status"], row["last_error"], row["last_run_at"], count))
        return result

    def clips_for_project(self, project_id: int) -> list[ClipRecord]:
        rows = self._db.query("SELECT * FROM clips WHERE project_id = ? ORDER BY start_sec", (project_id,))
        return [ClipRecord(Path(r["clip_path"]), Path(r["json_path"]) if r["json_path"] else None, r["title"], r["viral_score"], r["start_sec"], r["end_sec"]) for r in rows]

    def runs_for_project(self, project_id: int) -> list[dict]:
        return [dict(r) for r in self._db.query("SELECT * FROM runs WHERE project_id = ? ORDER BY started_at DESC", (project_id,))]
