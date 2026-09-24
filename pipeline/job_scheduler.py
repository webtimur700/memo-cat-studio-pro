"""Очередь пакетной обработки без привязки к Qt: сколько видео идёт одновременно.

Раньше каждое добавленное видео сразу получало свой поток — 100 видео давали 100
одновременных пайплайнов. Теперь одновременно работает не больше max_concurrent задач
(по умолчанию 1), остальные ждут в очереди FIFO. Класс однопоточный — его зовёт UI-поток;
запуск самой задачи (start_job) и реакция на пустую очередь (on_idle) — колбэки.
Очередь не ограничена по длине: deque, все операции O(1).
"""

from __future__ import annotations

from collections import deque
from typing import Callable


class JobScheduler:
    def __init__(
        self,
        max_concurrent: int,
        start_job: Callable[[str], None],
        on_idle: Callable[[], None] | None = None,
    ) -> None:
        self._max_concurrent = max(1, max_concurrent)
        self._start_job = start_job
        self._on_idle = on_idle
        self._waiting: deque[str] = deque()
        self._running: set[str] = set()

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def is_idle(self) -> bool:
        return not self._running and not self._waiting

    def submit(self, job_id: str) -> None:
        self._waiting.append(job_id)
        self._pump()

    def job_done(self, job_id: str) -> None:
        """Задача завершилась (успешно или с ошибкой): освободить слот и запустить следующую."""
        self._running.discard(job_id)
        self._pump()
        if self.is_idle and self._on_idle is not None:
            self._on_idle()

    def set_max_concurrent(self, value: int) -> None:
        """Новое значение из настроек: уже идущие задачи не прерываются, лишние ждут своей очереди."""
        self._max_concurrent = max(1, value)
        self._pump()

    def _pump(self) -> None:
        while self._waiting and len(self._running) < self._max_concurrent:
            job_id = self._waiting.popleft()
            self._running.add(job_id)
            try:
                self._start_job(job_id)
            except Exception:
                self._running.discard(job_id)
                raise
