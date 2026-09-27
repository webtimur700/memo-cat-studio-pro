"""Какие модели недавно не ответили (таймаут/зависание) — их селектор ставит ниже в рейтинге.

Пример: Qwen3.8-27B зависала на отдельных кадрах (запрос не завершался за 900 с). Такая модель
не исключается совсем (если больше ничего не помещается в память, она всё ещё лучше, чем ничего),
но идёт после всех моделей без сбоев. Штраф снимается успешным ответом или через PENALTY_TTL_SEC —
таймаут мог быть из-за состояния системы (например, почти полный своп), а не модели.

Хранится в JSON-файле рядом с базой (переживает перезапуск); без пути — только в памяти (тесты, скрипты).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from loguru import logger

PENALTY_TTL_SEC = 3 * 24 * 3600


class ModelHealth:
    def __init__(self, path: Path | None = None, now=time.time, ttl_sec: float = PENALTY_TTL_SEC) -> None:
        self._path = path
        self._now = now
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._failures: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("LLM: не прочитан {} ({}) — штрафы моделей сброшены", self._path, exc)
            return {}

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._failures, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("LLM: не записан {}: {}", self._path, exc)

    def penalized(self) -> frozenset[str]:
        """Модели со сбоем за последние PENALTY_TTL_SEC."""
        now = self._now()
        with self._lock:
            return frozenset(k for k, v in self._failures.items() if now - float(v.get("at", 0)) < self._ttl_sec)

    def record_failure(self, model_key: str, reason: str) -> None:
        with self._lock:
            entry = self._failures.get(model_key, {})
            self._failures[model_key] = {"at": self._now(), "count": int(entry.get("count", 0)) + 1, "reason": reason}
            self._save()
        logger.warning("LLM: {} понижена в рейтинге: {}", model_key, reason)

    def record_success(self, model_key: str) -> None:
        with self._lock:
            if self._failures.pop(model_key, None) is None:
                return
            self._save()
        logger.info("LLM: {} ответила — штраф в рейтинге снят", model_key)
