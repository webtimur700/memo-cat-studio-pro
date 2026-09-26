"""A/B: тексты клипа тремя запросами (как раньше) и одним JSON-запросом (structured output).

    python scripts/benchmark_llm_combined.py --models google/gemma-4-26b-a4b [--modes text image]
                                             [--out docs/llm_combined_benchmark_raw.json]

Идёт через production-код (llm/content_generator.generate_clip_content) на входах docs/llm_benchmark_inputs/inputs_v2.json.
Для каждой модели: выгрузка всего, загрузка REST с контекстом приложения (8192), прогрев одним запросом (первый вызов
после загрузки медленнее из-за подгрузки весов с диска), затем каждый вход в обоих вариантах, порядок чередуется,
чтобы дрейф скорости не доставался одному варианту. Результаты дописываются в JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_llm import MemorySampler, gpu_mib, meminfo_available_mib  # noqa: E402
from llm import lm_studio_api as api  # noqa: E402
from llm.content_generator import generate_clip_content  # noqa: E402
from llm.lm_studio_provider import LMStudioConfig, LMStudioProvider  # noqa: E402

BASE_URL = "http://localhost:1234/v1"
INPUTS = ROOT / "docs" / "llm_benchmark_inputs" / "inputs_v2.json"
CONTEXT_LENGTH = 8192


class _UsageSum:
    """Обёртка над провайдером: суммирует токены всех запросов клипа (last_usage хранит только последний)."""

    def __init__(self, provider: LMStudioProvider) -> None:
        self._p = provider
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __getattr__(self, name):
        attr = getattr(self._p, name)
        if name not in ("complete", "complete_json"):
            return attr

        def wrapper(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            finally:
                usage = self._p.last_usage or {}
                self.prompt_tokens += usage.get("prompt_tokens") or 0
                self.completion_tokens += usage.get("completion_tokens") or 0

        return wrapper


def run_case(provider: LMStudioProvider, transcript: str, image: bytes | None, combined: bool) -> dict:
    counted = _UsageSum(provider)
    started = time.time()
    content = generate_clip_content(counted, transcript, image_jpeg=image, combined=combined)
    return {
        "sec": round(time.time() - started, 1),
        "requests": content.requests,
        "prompt_tokens": counted.prompt_tokens,
        "completion_tokens": counted.completion_tokens,
        "errors": list(content.errors),
        "used_image": content.used_image,
        "titles": list(content.titles),
        "description": content.description,
        "hashtags": list(content.hashtags),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["text", "image"], choices=["text", "image"])
    parser.add_argument("--out", default=str(ROOT / "docs" / "llm_combined_benchmark_raw.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    inputs = json.loads(INPUTS.read_text())

    known = {m.key: m for m in api.list_models(BASE_URL) if m.is_chat_model}
    for key in args.models:
        info = known[key]
        entry = results.setdefault(key, {"cases": {}})
        api.unload_all(BASE_URL)
        time.sleep(3)
        baseline = meminfo_available_mib()
        t0 = time.time()
        api.load_model(BASE_URL, key, context_length=CONTEXT_LENGTH, parallel=1)
        entry["load_sec"] = round(time.time() - t0, 1)
        entry["reasoning_effort"] = "none" if info.supports_reasoning_control else None
        provider = LMStudioProvider(
            LMStudioConfig(base_url=BASE_URL, timeout_sec=900, model_override=key, reasoning_effort=entry["reasoning_effort"])
        )
        print(f"[{key}] загружена за {entry['load_sec']} с", flush=True)
        provider.complete("Отвечай одним словом.", "Привет", max_tokens=8)   # прогрев

        with MemorySampler() as mem:
            for n, item in enumerate(inputs):
                for mode in args.modes:
                    image = (INPUTS.parent / item["image"]).read_bytes() if mode == "image" else None
                    order = [False, True] if n % 2 == 0 else [True, False]   # separate / combined, порядок чередуется
                    for combined in order:
                        name = f"{mode}|{item['id']}|{'combined' if combined else 'separate'}"
                        if name in entry["cases"]:
                            continue
                        result = run_case(provider, item["transcript"], image, combined)
                        entry["cases"][name] = result
                        print(
                            f"[{key}] {name:34s} {result['sec']:6.1f}s  req={result['requests']} tok={result['completion_tokens']} "
                            f"titles={len(result['titles'])} tags={len(result['hashtags'])} err={result['errors']}",
                            flush=True,
                        )
                        entry["peak_available_drop_mib"] = round(baseline - mem.min_available)
                        entry["peak_gtt_mib"] = round(mem.max_gtt)
                        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    api.unload_all(BASE_URL)


if __name__ == "__main__":
    main()
