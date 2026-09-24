"""Замер LLM-моделей LM Studio на реальных промптах приложения.

    python scripts/benchmark_llm.py [--models KEY ...] [--modes none low default]
                                    [--out docs/llm_benchmark_raw.json]

Для каждой модели: выгрузить всё, замерить память, загрузить через REST с
ограниченным контекстом, прогнать заголовки/описание/хештеги на входах из
docs/llm_benchmark_inputs/inputs.json в разных режимах рассуждения. Результаты
дописываются в JSON по мере готовности (можно прервать и продолжить).

Режимы: none — reasoning_effort=none (рассуждения выключены), low — ограничены,
default — параметр не передаётся (как настроена модель).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import lm_studio_api as api  # noqa: E402
from llm.content_generator import build_clip_description  # noqa: E402
from llm.lm_studio_provider import LLMRequestError, LMStudioConfig, LMStudioProvider  # noqa: E402
from llm.prompts.description_prompt import generate_description  # noqa: E402
from llm.prompts.hashtags_prompt import generate_hashtags  # noqa: E402
from llm.prompts.titles_prompt import generate_titles  # noqa: E402

BASE_URL = "http://localhost:1234/v1"
INPUTS = ROOT / "docs" / "llm_benchmark_inputs" / "inputs.json"
CONTEXT_LENGTH = 16384
REASONING_MODE_INPUT_IDS = {"m990", "m60"}   # в долгих режимах (low, default) — только 2 входа


def meminfo_available_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return 0.0


def gpu_mib(name: str) -> float:
    for card in Path("/sys/class/drm").glob("card[0-9]/device"):
        f = card / name
        if f.exists():
            return int(f.read_text()) / 1048576
    return 0.0


class MemorySampler:
    """Фоновый замер: минимум MemAvailable и максимум GTT/VRAM за интервал."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.min_available = meminfo_available_mib()
        self.max_gtt = gpu_mib("mem_info_gtt_used")
        self.max_vram = gpu_mib("mem_info_vram_used")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.min_available = min(self.min_available, meminfo_available_mib())
            self.max_gtt = max(self.max_gtt, gpu_mib("mem_info_gtt_used"))
            self.max_vram = max(self.max_vram, gpu_mib("mem_info_vram_used"))
            self._stop.wait(0.5)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()


def analyze(titles: list[str], description: str, hashtags: list[str], raw_titles: str) -> dict:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", " ".join(titles))
    cyr = sum(1 for c in letters if "А" <= c <= "я" or c in "Ёё")
    return {
        "titles_count": len(titles),
        "titles_exactly_10": len(titles) == 10,
        "titles_max_len": max((len(t) for t in titles), default=0),
        "titles_over_60": sum(len(t) > 60 for t in titles),
        "titles_unique": len(set(t.lower() for t in titles)),
        "titles_cyrillic_ratio": round(cyr / len(letters), 2) if letters else 0.0,
        "description_len": len(description),
        "hashtags_count": len(hashtags),
        "hashtags_ok": bool(hashtags) and all(h.startswith("#") for h in hashtags),
    }


def run_case(provider: LMStudioProvider, text: str) -> dict:
    """Один клип: три запроса как в production. Возвращает тайминги, токены, ответы."""
    record: dict = {"steps": {}}
    total_start = time.time()
    outputs: dict = {}
    for name, fn in (
        ("titles", lambda: generate_titles(provider, text, count=10)),
        ("description", lambda: generate_description(provider, text)),
        ("hashtags", lambda: generate_hashtags(provider, text, max_count=30)),
    ):
        t0 = time.time()
        try:
            outputs[name] = fn()
            error = None
        except LLMRequestError as exc:
            outputs[name] = [] if name != "description" else ""
            error = str(exc)[:200]
        usage = provider.last_usage or {}
        details = usage.get("completion_tokens_details") or {}
        record["steps"][name] = {
            "sec": round(time.time() - t0, 1),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "error": error,
        }
    record["total_sec"] = round(time.time() - total_start, 1)
    record["outputs"] = outputs
    record["metrics"] = analyze(outputs["titles"], outputs["description"], outputs["hashtags"], "")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--modes", nargs="*", default=["none", "low", "default"])
    parser.add_argument("--out", default=str(ROOT / "docs" / "llm_benchmark_raw.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    inputs = json.loads(INPUTS.read_text())

    models = [m for m in api.list_models(BASE_URL) if m.is_chat_model]
    if args.models:
        models = [m for m in models if m.key in args.models]

    for model in models:
        entry = results.setdefault(model.key, {"cases": {}})
        api.unload_all(BASE_URL)
        time.sleep(3)
        baseline_avail = meminfo_available_mib()
        baseline_gtt = gpu_mib("mem_info_gtt_used")

        already_loaded = "load" in entry
        with MemorySampler() as mem_load:
            t0 = time.time()
            api.load_model(BASE_URL, model.key, context_length=CONTEXT_LENGTH, parallel=1)
            load_sec = time.time() - t0
        time.sleep(2)
        entry["load"] = {
            "load_sec": round(load_sec, 1),
            "context_length": CONTEXT_LENGTH,
            "available_drop_mib": round(baseline_avail - meminfo_available_mib()),
            "gtt_delta_mib": round(gpu_mib("mem_info_gtt_used") - baseline_gtt),
            "size_gib": round(model.size_gib or 0, 1),
        }
        print(f"[{model.key}] загружена за {load_sec:.0f} с, память: {entry['load']}", flush=True)

        with MemorySampler() as mem_run:
            for mode in args.modes:
                for item in inputs:
                    if mode != "none" and item["id"] not in REASONING_MODE_INPUT_IDS:
                        continue
                    case_key = f"{mode}|{item['id']}"
                    if case_key in entry["cases"]:
                        continue
                    effort = None if mode == "default" else mode
                    provider = LMStudioProvider(
                        LMStudioConfig(base_url=BASE_URL, timeout_sec=900, model_override=model.key, reasoning_effort=effort)
                    )
                    text = build_clip_description(item["transcript"])
                    result = run_case(provider, text)
                    entry["cases"][case_key] = result
                    m = result["metrics"]
                    print(
                        f"[{model.key}] {case_key:12s} {result['total_sec']:6.1f}s  titles={m['titles_count']} "
                        f"tags={m['hashtags_count']} desc={m['description_len']}  "
                        f"errors={[k for k, v in result['steps'].items() if v['error']]}",
                        flush=True,
                    )
                    entry["run_memory"] = {
                        "peak_available_drop_mib": round(baseline_avail - mem_run.min_available),
                        "peak_gtt_mib": round(mem_run.max_gtt),
                        "peak_vram_mib": round(mem_run.max_vram),
                    }
                    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    api.unload_all(BASE_URL)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
