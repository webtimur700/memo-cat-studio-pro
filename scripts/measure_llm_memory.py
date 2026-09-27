"""Сколько памяти реально занимает LLM в LM Studio: падение MemAvailable после загрузки
(столько вернёт выгрузка) и пик при единичном запросе текстов клипа с кадром.

Нужен для профилей в llm/model_selector.py (peak_mib / loaded_mib): оценка по размеру GGUF
ошибается в обе стороны — у плотной модели пик выше, у MoE ниже.

    python scripts/measure_llm_memory.py --models google/gemma-4-26b-a4b qwen/qwen3.8-27b
Перед замером все модели LM Studio выгружаются; результат — JSON в stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import lm_studio_api as api  # noqa: E402
from llm.lm_studio_provider import LMStudioConfig, LMStudioProvider  # noqa: E402
from llm.model_selector import read_mem_available_mib  # noqa: E402
from llm.prompts.clip_content_prompt import (  # noqa: E402
    CLIP_CONTENT_SCHEMA,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    build_clip_content_prompt,
)

BASE_URL = "http://localhost:1234/v1"
INPUTS = Path(__file__).resolve().parent.parent / "docs" / "llm_benchmark_inputs"


class MinSampler:
    """Минимум MemAvailable в фоне (0.25 с)."""

    def __init__(self) -> None:
        self.min_mib = read_mem_available_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self.min_mib = min(self.min_mib, read_mem_available_mib())

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self.min_mib


def settle(seconds: float = 5.0) -> float:
    time.sleep(seconds)
    return read_mem_available_mib()


def measure(key: str, context_length: int, input_id: str) -> dict:
    api.unload_all(BASE_URL)
    before = settle()
    sampler = MinSampler()
    started = time.perf_counter()
    instance = api.load_model(BASE_URL, key, context_length=context_length)
    load_sec = time.perf_counter() - started
    after_load = settle()

    item = next(i for i in json.loads((INPUTS / "inputs_v2.json").read_text()) if i["id"] == input_id)
    image = (INPUTS / Path(item["image"]).name).read_bytes()
    provider = LMStudioProvider(LMStudioConfig(base_url=BASE_URL, timeout_sec=600, model_override=instance, reasoning_effort="none"))
    started = time.perf_counter()
    error = None
    try:
        provider.complete_json(
            SYSTEM_PROMPT, build_clip_content_prompt(item["transcript"], has_image=True), CLIP_CONTENT_SCHEMA,
            max_tokens=MAX_TOKENS, images=[image], schema_name="clip_content",
        )
    except Exception as exc:   # замер всё равно полезен: пик до сбоя
        error = str(exc)
    request_sec = time.perf_counter() - started
    minimum = sampler.stop()
    api.unload_model(BASE_URL, instance)
    after_unload = settle()
    return {
        "model": key, "context_length": context_length, "input": input_id,
        "mem_available_before_gib": round(before / 1024, 2),
        "loaded_drop_gib": round((before - after_load) / 1024, 2),
        "peak_drop_gib": round((before - minimum) / 1024, 2),
        "returned_by_unload_gib": round((after_unload - after_load) / 1024, 2),
        "load_sec": round(load_sec, 1), "request_sec": round(request_sec, 1), "error": error,
        "usage": provider.last_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--context", type=int, default=LMStudioConfig().context_length)
    parser.add_argument("--input", default="m150", help="id входа из docs/llm_benchmark_inputs/inputs_v2.json")
    args = parser.parse_args()
    results = []
    for key in args.models:
        result = measure(key, args.context, args.input)
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
        results.append(result)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
