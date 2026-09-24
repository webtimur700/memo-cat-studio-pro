"""Сводные таблицы Markdown из сырых результатов бенчмарков.

    python scripts/benchmark_report.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _cases(entry: dict, mode: str) -> list[dict]:
    return [c for k, c in entry["cases"].items() if k.split("|")[0] == mode]


def llm_tables(raw: dict, title: str) -> str:
    lines = [f"#### {title}", "",
             "| Модель | Режим | Клипов | сек/клип (мин–макс) | 10 заголовков | хештегов (ср.) | все с # | ошибки | токенов/клип (из них рассуждения) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, entry in raw.items():
        modes = sorted({k.split("|")[0] for k in entry["cases"]}, key=lambda m: ["none", "none_img", "low", "default"].index(m))
        for mode in modes:
            cases = _cases(entry, mode)
            secs = [c["total_sec"] for c in cases]
            tags = [c["metrics"]["hashtags_count"] for c in cases]
            errs = sum(1 for c in cases for s in c["steps"].values() if s["error"])
            tokens = [sum((s["completion_tokens"] or 0) for s in c["steps"].values()) for c in cases]
            reasoning = [sum((s["reasoning_tokens"] or 0) for s in c["steps"].values()) for c in cases]
            lines.append(
                f"| {model} | {mode} | {len(cases)} | {st.mean(secs):.0f} ({min(secs):.0f}–{max(secs):.0f}) | "
                f"{sum(c['metrics']['titles_exactly_10'] for c in cases)}/{len(cases)} | {st.mean(tags):.0f} | "
                f"{sum(c['metrics']['hashtags_ok'] for c in cases)}/{len(cases)} | {errs} | "
                f"{st.mean(tokens):.0f} ({st.mean(reasoning):.0f}) |"
            )
    return "\n".join(lines)


def memory_table(raw: dict) -> str:
    lines = ["| Модель | Размер GGUF | Загрузка, с | MemAvailable после загрузки | GTT после загрузки | Пик MemAvailable при генерации | Пик GTT |",
             "|---|---|---|---|---|---|---|"]
    for model, entry in raw.items():
        load, run = entry["load"], entry.get("run_memory", {})
        lines.append(
            f"| {model} | {load['size_gib']} ГиБ | {load['load_sec']:.0f} | −{load['available_drop_mib'] / 1024:.1f} ГиБ | "
            f"+{load['gtt_delta_mib'] / 1024:.1f} ГиБ | −{run.get('peak_available_drop_mib', 0) / 1024:.1f} ГиБ | "
            f"{run.get('peak_gtt_mib', 0) / 1024:.1f} ГиБ |"
        )
    return "\n".join(lines)


def whisper_table(raw: dict) -> str:
    lines = ["| Конфигурация | Загрузка, с | Σ время окон, с | макс. окно, с | ср. logprob | ср. WER к эталону |", "|---|---|---|---|---|---|"]
    for name, entry in raw.items():
        windows = entry["windows"].values()
        secs = [w["sec"] for w in windows]
        lps = [w["avg_logprob"] for w in windows if w["avg_logprob"] is not None]
        wers = [w["wer_vs_reference"] for w in windows if "wer_vs_reference" in w]
        lines.append(
            f"| {name} | {entry.get('load_sec', 0)} | {sum(secs):.0f} | {max(secs):.0f} | "
            f"{st.mean(lps):.2f} | {'—' if not wers else f'{st.mean(wers):.2f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parts = []
    for filename, title in (("llm_benchmark_raw.json", "Раунд 1 (расшифровки с принудительным `ru`, промпты без защиты)"),
                            ("llm_benchmark_v2_raw.json", "Раунд 2 (расшифровки с автоопределением языка, промпты с защитой)")):
        path = DOCS / filename
        if path.exists():
            raw = json.loads(path.read_text())
            parts.append(llm_tables(raw, title))
            if filename == "llm_benchmark_raw.json":
                parts.append("\n#### Память\n\n" + memory_table(raw))
    whisper_path = DOCS / "whisper_benchmark_raw.json"
    if whisper_path.exists():
        parts.append("\n#### Whisper\n\n" + whisper_table(json.loads(whisper_path.read_text())))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
