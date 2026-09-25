"""Замер времени стадий пайплайна: накопительный секундомер по именам (потокобезопасный).

    with stage("export.encode"): ...
    snapshot()  ->  {"export.encode": (секунды, вызовов), ...}

Накладные расходы — один perf_counter на вход/выход, поэтому замеры включены всегда;
`scripts/profile_pipeline.py` печатает итоговую таблицу, в лог пишется по одной строке на стадию.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

_lock = threading.Lock()
_totals: dict[str, float] = defaultdict(float)
_calls: dict[str, int] = defaultdict(int)


@contextmanager
def stage(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        with _lock:
            _totals[name] += elapsed
            _calls[name] += 1
        logger.debug("stage {}: {:.2f} с", name, elapsed)


def snapshot() -> dict[str, tuple[float, int]]:
    with _lock:
        return {name: (_totals[name], _calls[name]) for name in _totals}


def reset() -> None:
    with _lock:
        _totals.clear()
        _calls.clear()
