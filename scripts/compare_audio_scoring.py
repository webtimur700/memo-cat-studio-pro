"""Как звуковые события меняют отбор моментов: один и тот же скан видео без audio_event и с ним.

    python scripts/compare_audio_scoring.py ~/Видео/myvideo.mp4 [--out result.json]

Паузы речи (Whisper) не используются — сравнивается только влияние звуковых событий на оценки и отбор.
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
    runner = PipelineRunner(models_dir=Path("models"), output_dir=Path("/tmp/audio_cmp_tmp"), shared_models=shared)
    duration = IngestionService().ingest(video).duration_sec
    scenes = runner._detect_scenes_safely(video)
    boundaries = sorted({s.start_sec for s in scenes if s.start_sec > 0} | {s.end_sec for s in scenes if s.end_sec < duration})
    detector = _DetectorHandle(shared.detector())

    t0 = time.time()
    timeline = runner._analyze_audio_events(video)
    print(f"YAMNet: {time.time() - t0:.1f} с на {duration:.0f} с видео, окон звука: {timeline.frame_scores.size if timeline else 0}")
    if timeline is None:
        sys.exit("нет YAMNet или звука")

    print("скан без звуковых событий...", flush=True)
    windows_off = runner._scan_windows(video, duration, detector, scenes, None, settings.viral_score)
    print("скан со звуковыми событиями...", flush=True)
    windows_on = runner._scan_windows(video, duration, detector, scenes, timeline, settings.viral_score)

    moments_off = select_moments(windows_off, settings, duration, scene_boundaries=boundaries)
    moments_on = select_moments(windows_on, settings, duration, scene_boundaries=boundaries)

    def stats(ws):
        s = sorted(w.viral_score for w in ws)
        return f"min={s[0]} медиана={s[len(s) // 2]} p90={s[int(len(s) * 0.9)]} max={s[-1]}"

    print(f"\nокон: {len(windows_on)};  score без звука: {stats(windows_off)};  со звуком: {stats(windows_on)}")
    events = [(w.start_sec, timeline.score_between(w.start_sec, w.end_sec), timeline.dominant_class_between(w.start_sec, w.end_sec)) for w in windows_on]
    loud = [e for e in events if e[1] >= 0.5]
    print(f"окон с явным звуковым событием (>=0.5): {len(loud)} из {len(events)}")
    by_class: dict[str, int] = {}
    for _, _, cls in loud:
        by_class[cls] = by_class.get(cls, 0) + 1
    print("  по классам:", dict(sorted(by_class.items(), key=lambda kv: -kv[1])))

    def show(title, moments):
        print(f"\n{title}: {len(moments)} моментов")
        for m in moments:
            ev = timeline.score_between(m.start_sec, m.end_sec)
            print(f"  {fmt(m.start_sec)} - {fmt(m.end_sec)} ({m.duration_sec:4.1f} с) score {m.viral_score:3d}  звук {ev:.2f} {timeline.dominant_class_between(m.start_sec, m.end_sec)}")

    show("БЕЗ звуковых событий", moments_off)
    show("СО звуковыми событиями", moments_on)

    def overlap(a, b):
        return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))

    same = sum(1 for a in moments_on if any(overlap(a, b) > 0.5 * a.duration_sec for b in moments_off))
    print(f"\nмоментов «со звуком», совпавших с выбором «без звука» (перекрытие > 50%): {same} из {len(moments_on)}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "off": [(m.start_sec, m.end_sec, m.viral_score) for m in moments_off],
            "on": [(m.start_sec, m.end_sec, m.viral_score) for m in moments_on],
            "loud_windows": loud,
        }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
