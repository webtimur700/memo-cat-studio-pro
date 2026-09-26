"""Замок с приоритетом: срочные захваты (`urgent()`) обходят ожидающих в обычной очереди (`background()`).

Нужен для LM Studio: запросы идут строго по одному, но перевод субтитров нужен кодированию клипа сразу, а тексты
клипа считаются в фоне. Обычный threading.Lock не гарантирует порядка: фоновый поток, закончив один запрос, тут же
берёт замок для следующего, и перевод мог ждать все накопившиеся тексты подряд (замер: 104 с вместо ~10).
Идущий запрос не прерывается — срочный захват ждёт только его.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class PriorityLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._urgent_waiting = 0

    @contextmanager
    def urgent(self):
        with self._cond:
            self._urgent_waiting += 1
            try:
                while self._busy:
                    self._cond.wait()
            finally:
                self._urgent_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def background(self):
        with self._cond:
            while self._busy or self._urgent_waiting:
                self._cond.wait()
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self._cond:
            self._busy = False
            self._cond.notify_all()
