"""Локальная база SQLite (пункт ТЗ «ЛОКАЛЬНАЯ БАЗА»): настройки, проекты и история обработки.

Файл по умолчанию — database/memo_cat.db (.env: DB_PATH). Схема версионируется через PRAGMA user_version:
миграции применяются по порядку при открытии, поэтому старая база обновляется сама. Соединение одно на
приложение, доступ сериализован замком (пишет UI-поток, но читать могут и фоновые).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

MIGRATIONS: tuple[str, ...] = (
    # 1: настройки — по строке на изменённое пользователем поле (значение — JSON), проекты, запуски, клипы
    """
    CREATE TABLE settings (
        section TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (section, key)
    );
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        added_at REAL NOT NULL,
        last_status TEXT NOT NULL DEFAULT 'queued',
        last_error TEXT,
        last_run_at REAL
    );
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        started_at REAL NOT NULL,
        finished_at REAL,
        status TEXT NOT NULL DEFAULT 'running',
        moments INTEGER NOT NULL DEFAULT 0,
        error TEXT
    );
    CREATE TABLE clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        clip_path TEXT NOT NULL UNIQUE,
        json_path TEXT,
        title TEXT NOT NULL DEFAULT '',
        viral_score INTEGER NOT NULL DEFAULT 0,
        start_sec REAL NOT NULL DEFAULT 0,
        end_sec REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE INDEX idx_runs_project ON runs(project_id);
    CREATE INDEX idx_clips_project ON clips(project_id);
    """,
)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for number, script in enumerate(MIGRATIONS, start=1):
                if number > version:
                    logger.info("БД {}: миграция {}", self.path.name, number)
                    self._conn.executescript(script)
                    self._conn.execute(f"PRAGMA user_version = {number}")
            self._conn.commit()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cursor

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def transaction(self, statements: list[tuple[str, tuple]]) -> None:
        """Несколько записей одной транзакцией: либо все, либо ни одной."""
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
