"""Замер ВСЕГО пайплайна с реальной LLM: время на клип и пик памяти системы.

    python scripts/benchmark_pipeline.py video.mp4 --models KEY [KEY ...] [--no-vision]
                                         [--out docs/pipeline_benchmark_raw.json]

На каждую модель: выгрузить всё, прогнать PipelineRunner (Whisper, YOLO, ffmpeg, LLM
одновременно живут в памяти), параллельно снимать MemAvailable, GTT iGPU и RSS
процессов (пайплайн, ffmpeg, LM Studio). Клипы пишутся во временную папку.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.entities.settings import UserSettings  # noqa: E402
from llm import lm_studio_api as api  # noqa: E402
from llm.lm_studio_provider import LMStudioConfig  # noqa: E402
from llm.managed_provider import ManagedLMStudio  # noqa: E402
from pipeline.pipeline_runner import PipelineRunner  # noqa: E402

BASE_URL = "http://localhost:1234/v1"


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) // 1024   # МиБ
    return values


def gpu_mib(name: str) -> int:
    for card in Path("/sys/class/drm").glob("card[0-9]/device"):
        f = card / name
        if f.exists():
            return int(f.read_text()) // 1048576
    return 0


def rss_by_group() -> dict[str, int]:
    """RSS (МиБ) по группам: пайплайн (python), ffmpeg, LM Studio."""
    groups = {"python": 0, "ffmpeg": 0, "lm_studio": 0}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            rss_pages = int((proc / "statm").read_text().split()[1])
        except (OSError, ValueError, IndexError):
            continue
        rss_mib = rss_pages * 4096 // 1048576
        if "benchmark_pipeline.py" in cmdline and "python" in cmdline:
            groups["python"] += rss_mib
        elif cmdline.startswith("ffmpeg") or "/ffmpeg " in cmdline:
            groups["ffmpeg"] += rss_mib
        elif "lm-studio" in cmdline or "llmster" in cmdline or "llama" in cmdline:
            groups["lm_studio"] += rss_mib
    return groups


class Sampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.min_available = meminfo()["MemAvailable"]
        self.max_gtt = gpu_mib("mem_info_gtt_used")
        self.peak_rss = {"python": 0, "ffmpeg": 0, "lm_studio": 0}
        self.peak_swap_used = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            info = meminfo()
            self.min_available = min(self.min_available, info["MemAvailable"])
            self.peak_swap_used = max(self.peak_swap_used, info["SwapTotal"] - info["SwapFree"])
            self.max_gtt = max(self.max_gtt, gpu_mib("mem_info_gtt_used"))
            for group, value in rss_by_group().items():
                self.peak_rss[group] = max(self.peak_rss[group], value)
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "docs" / "pipeline_benchmark_raw.json"))
    args = parser.parse_args()

    video = Path(args.video).expanduser()
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    settings = UserSettings()
    from dataclasses import replace

    settings = replace(settings, viral_score=replace(settings.viral_score, queue_threshold=0))

    for model_key in args.models:
        api.unload_all(BASE_URL)
        time.sleep(3)
        baseline = meminfo()["MemAvailable"]
        baseline_gtt = gpu_mib("mem_info_gtt_used")

        llm = ManagedLMStudio(LMStudioConfig(base_url=BASE_URL, model_override=model_key, use_vision=not args.no_vision, timeout_sec=900))
        stage_times: dict[str, float] = {}
        t0 = time.time()

        def on_progress(stage: str, data: dict) -> None:
            stage_times.setdefault(stage, time.time() - t0)

        with tempfile.TemporaryDirectory() as tmp, Sampler() as sampler:
            llm.start_loading_in_background()
            runner = PipelineRunner(models_dir=ROOT / "models", output_dir=Path(tmp), llm_provider=llm)
            clips = runner.process_video(video, settings, progress=on_progress)
            total = time.time() - t0
            llm_model_used = llm.selection.model_key if llm.selection else None
            vision = llm.supports_vision
            with_titles = sum(1 for c in clips if c.titles)
        llm.release()

        key = model_key + ("" if not args.no_vision else "|text")
        results[key] = {
            "video": video.name, "clips": len(clips), "clips_with_llm_titles": with_titles,
            "total_sec": round(total), "sec_per_clip": round(total / max(1, len(clips))),
            "stage_start_sec": {k: round(v) for k, v in stage_times.items()},
            "vision_used": vision, "model": llm_model_used,
            "peak_available_drop_mib": baseline - sampler.min_available,
            "min_available_mib": sampler.min_available,
            "peak_gtt_delta_mib": sampler.max_gtt - baseline_gtt,
            "peak_rss_mib": sampler.peak_rss, "peak_swap_used_mib": sampler.peak_swap_used,
        }
        print(f"[{key}] {json.dumps(results[key], ensure_ascii=False)}", flush=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
