"""Как прыжки и падения меняют отбор моментов: один скан видео, оценка без бонуса и с ним.

    python scripts/compare_motion_scoring.py ~/Видео/myvideo.mp4 [--out result.json]

Звуковые события включены в обоих вариантах (это уже часть Viral Score), паузы речи не используются.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entities.settings import UserSettings  # noqa: E402
from cutting.clip_selector_service import select_moments  # noqa: E402
from pipeline.pipeline_runner import PipelineRunner, _DetectorHandle  # noqa: E402
from pipeline.shared_models import SharedModels  # noqa: E402
from video.ingestion_service import IngestionService  # noqa: E402


def fmt(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    video = Path(args.video).expanduser()
    settings = UserSettings()
    shared = SharedModels(Path("models"))
    runner = PipelineRunner(models_dir=Path("models"), output_dir=Path("/tmp/motion_cmp_tmp"), shared_models=shared)
    duration = IngestionService().ingest(video).duration_sec
    scenes = runner._detect_scenes_safely(video)
    boundaries = sorted({s.start_sec for s in scenes if s.start_sec > 0} | {s.end_sec for s in scenes if s.end_sec < duration})
    detector = _DetectorHandle(shared.detector())
    audio = runner._analyze_audio_events(video)

    t0 = time.time()
    base = runner._scan_windows(video, duration, detector, scenes, audio, settings.viral_score)
    t_scan = time.time() - t0
    t0 = time.time()
    with_motion = runner._add_motion_events(video, base, detector, scenes, settings.viral_score)
    t_motion = time.time() - t0

    changed = [(a, b) for a, b in zip(base, with_motion) if a.viral_score != b.viral_score]
    print(f"скан 1 кадр/с: {t_scan:.0f} с; плотный анализ движения: {t_motion:.0f} с; окон {len(base)}, оценка выросла у {len(changed)}")
    for a, b in changed:
        print(f"  {fmt(a.start_sec)}  {a.viral_score:3d} -> {b.viral_score:3d}   {b.inputs.motion_events_detail}")

    moments_a = select_moments(base, settings, duration, scene_boundaries=boundaries)
    moments_b = select_moments(with_motion, settings, duration, scene_boundaries=boundaries)

    def show(title, moments):
        print(f"\n{title}: {len(moments)} моментов")
        for m in moments:
            inside = [w for w in with_motion if w.start_sec < m.end_sec and w.end_sec > m.start_sec and w.inputs and w.inputs.motion_events > 0]
            what = "; ".join(f"{fmt(w.start_sec)} {w.inputs.motion_events_detail}" for w in inside)
            print(f"  {fmt(m.start_sec)} - {fmt(m.end_sec)} ({m.duration_sec:4.1f} с) score {m.viral_score:3d}  {what}")

    show("БЕЗ прыжков и падений", moments_a)
    show("С прыжками и падениями", moments_b)

    def overlap(x, y):
        return max(0.0, min(x.end_sec, y.end_sec) - max(x.start_sec, y.start_sec))

    same = sum(1 for m in moments_b if any(overlap(m, o) > 0.5 * m.duration_sec for o in moments_a))
    print(f"\nмоментов «с бонусом», совпавших с выбором «без» (перекрытие > 50%): {same} из {len(moments_b)}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "without": [(m.start_sec, m.end_sec, m.viral_score) for m in moments_a],
            "with": [(m.start_sec, m.end_sec, m.viral_score) for m in moments_b],
            "changed": [(a.start_sec, a.viral_score, b.viral_score, b.inputs.motion_events_detail) for a, b in changed],
        }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
