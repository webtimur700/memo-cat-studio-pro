"""Оценки окон и результат отбора моментов на видео: гистограмма + таблица.

    python scripts/moment_distribution.py video.mp4 [--cache scores.json]

Оценки окон кэшируются в JSON (скан видео — самая долгая часть), поэтому
"до" и "после" изменения алгоритма отбора сравниваются на одних и тех же окнах.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entities.settings import UserSettings  # noqa: E402
from cutting.clip_selector_service import WindowScore, select_moments  # noqa: E402
from pipeline.pipeline_runner import PipelineRunner, _try_load_yolo  # noqa: E402
from video.ingestion_service import IngestionService  # noqa: E402


def scan(video: Path) -> tuple[list[WindowScore], float, list[float]]:
    runner = PipelineRunner(models_dir=Path("models"))
    duration = IngestionService().ingest(video).duration_sec
    scenes = runner._detect_scenes_safely(video)
    windows = runner._scan_windows(video, duration, _try_load_yolo(Path("models")), scenes)
    boundaries = sorted({s.start_sec for s in scenes if s.start_sec > 0} | {s.end_sec for s in scenes if s.end_sec < duration})
    return windows, duration, boundaries


def load_or_scan(video: Path, cache: Path) -> tuple[list[WindowScore], float, list[float]]:
    if cache.exists():
        data = json.loads(cache.read_text())
        windows = [WindowScore(w["start"], w["end"], w["score"], w["motion"]) for w in data["windows"]]
        return windows, data["duration"], data["scene_boundaries"]
    windows, duration, boundaries = scan(video)
    cache.write_text(json.dumps({
        "duration": duration, "scene_boundaries": boundaries,
        "windows": [{"start": w.start_sec, "end": w.end_sec, "score": w.viral_score, "motion": w.motion_intensity} for w in windows],
    }))
    return windows, duration, boundaries


def histogram(windows: list[WindowScore], selected: list[tuple[float, float]]) -> str:
    lines = []
    for w in windows:
        inside = any(a <= w.start_sec + 0.1 and w.end_sec - 0.1 <= b for a, b in selected)
        bar = "#" * (w.viral_score // 2)
        lines.append(f"{w.start_sec:6.0f}-{w.end_sec:<6.0f} {w.viral_score:3d} {bar:<50s} {'<== в клипе' if inside else ''}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--no-pauses", action="store_true", help="не искать паузы речи (без Whisper)")
    args = parser.parse_args()
    video = Path(args.video).expanduser()
    cache = Path(args.cache) if args.cache else video.with_suffix(".scores.json")
    windows, duration, boundaries = load_or_scan(video, cache)
    settings = UserSettings()
    runner = PipelineRunner(models_dir=Path("models"), output_dir=Path("/tmp/moment_distribution_tmp"))
    moments = select_moments(
        windows, settings, duration, scene_boundaries=boundaries,
        pause_finder=None if args.no_pauses else (lambda lo, hi: runner._find_speech_pauses(video, lo, hi, settings)),
    )
    scores = sorted(w.viral_score for w in windows)
    print(f"видео {duration:.0f} с, окон {len(windows)}, score: min={scores[0]} медиана={scores[len(scores)//2]} max={scores[-1]}")
    print(f"смен сцен: {len(boundaries)}")
    print(histogram(windows, [(m.start_sec, m.end_sec) for m in moments]))
    total = sum(m.duration_sec for m in moments)
    print(f"\nМоментов: {len(moments)}, суммарно {total:.0f} с = {100 * total / duration:.0f}% видео")
    for m in moments:
        print(f"  {m.start_sec:7.1f} - {m.end_sec:7.1f}  ({m.duration_sec:4.1f} с)  score {m.viral_score}")


if __name__ == "__main__":
    main()
