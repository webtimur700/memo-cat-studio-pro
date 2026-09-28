"""Регрессионная проверка на наборе тестовых видео: VFR, 4K, высокий битрейт, повороты, HDR.

    python scripts/regress_video_variants.py SOURCE WORK_DIR [--start 120 --duration 180] [--only имя.mp4 ...]
                                             [--real ролик.mp4 ...] [--json отчёт.json]

1. scripts/make_test_videos.py делает варианты из SOURCE в WORK_DIR/videos (готовые файлы не пересоздаются;
   видео в git не кладутся — WORK_DIR вне репозитория).
2. Каждый вариант обрабатывается пайплайном без LLM (scripts/profile_pipeline.py, отдельный процесс) в WORK_DIR/runs/<имя>.
3. Каждый вариант сверяется с контрольной CFR-копией (cfr_control.mp4):
   * моменты: доля моментов контроля, у которых в варианте есть момент с пересечением IoU ≥ 0.5 (не меньше MIN_MOMENT_MATCH);
   * синхронизация (метки: вспышка + писк): все метки в клипах найдены, внесённый рассинхрон |звук − видео| ≤ 45 мс, ошибка
     места видео и звука относительно исходника ≤ 50 мс;
   * ориентация: кадры клипа сравниваются с кадрами контрольного клипа в те же моменты исходника (корреляция сильно
     уменьшенных и размытых серых кадров — кроп-трекинг в разных прогонах сдвигает окно, резкие миниатюры это путают) и
     с тем же контролем, повёрнутым на 90°, 180° и 270°: в большинстве образцов каждой пары ближе всех должен быть
     неповёрнутый контроль;
   * выход: H.264 1080x1920, 30/1 fps, без тега поворота; у HDR — ещё и средняя насыщенность/контраст клипа не ниже
     HDR_MIN_RATIO от контроля (блёклые цвета без тонмаппинга).
4. --real: настоящие ролики (без меток): только выход, длины звука/видео и отсутствие ошибок.
Код возврата 1, если хоть одна проверка не прошла. Сводка — таблица в stdout и JSON (--json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import check_av_sync as sync  # noqa: E402
import make_test_videos as maker  # noqa: E402

MIN_MOMENT_IOU = 0.5
MIN_MOMENT_MATCH = 0.6
MAX_AV_OFFSET_MS = 45.0
MAX_POSITION_ERR_MS = 50.0
MIN_ORIENT_VOTES = 2 / 3
HDR_MIN_RATIO = 0.8
THUMB_W, THUMB_H = 18, 32


@dataclass
class Result:
    name: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def run_pipeline(video: Path, out: Path) -> None:
    if (out / "profile.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(SCRIPTS / "profile_pipeline.py"), str(video), "--keep", str(out),
                        "--json", str(out / "profile.json")], stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def clips(run_dir: Path) -> list[tuple[Path, float, float]]:
    """(клип, начало, конец момента в исходнике). mp4 без .json — экспорт упал на полпути, это не клип."""
    result = []
    for clip in sorted(run_dir.glob("*_moment*.mp4")):
        if not clip.with_suffix(".json").exists():
            continue
        meta = json.loads(clip.with_suffix(".json").read_text())
        result.append((clip, float(meta["start_sec"]), float(meta["end_sec"])))
    return result


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_moments(control: list[tuple[float, float]], variant: list[tuple[float, float]]) -> list[tuple[int, int, float]]:
    """Для каждого момента контроля — лучший по IoU момент варианта: (i контроля, j варианта или -1, IoU)."""
    pairs = []
    for i, c in enumerate(control):
        best = max(((j, iou(c, v)) for j, v in enumerate(variant)), key=lambda x: x[1], default=(-1, 0.0))
        pairs.append((i, best[0] if best[1] >= MIN_MOMENT_IOU else -1, round(best[1], 3)))
    return pairs


def thumb(clip: Path, at: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, at):.3f}", "-i", str(clip), "-frames:v", "1",
         "-vf", f"scale={THUMB_W}:{THUMB_H}:flags=area,gblur=sigma=1.5,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8)[: THUMB_W * THUMB_H].reshape(THUMB_H, THUMB_W).astype(np.float32)


def rotations(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Кадр и его повороты на 90/180/270° (повёрнутые на 90° приведены к тому же размеру: форма кадра тоже признак)."""
    import cv2

    return {0: frame, 180: frame[::-1, ::-1],
            90: cv2.resize(np.rot90(frame, 1), (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA),
            270: cv2.resize(np.rot90(frame, 3), (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def orientation(variant: tuple[Path, float, float], control: tuple[Path, float, float]) -> dict:
    """Три момента в общей части клипов (время исходника): какой поворот контроля ближе всего к кадру варианта."""
    lo, hi = max(variant[1], control[1]), min(variant[2], control[2])
    votes = []
    for k in (0.25, 0.5, 0.75):
        t = lo + (hi - lo) * k
        v = thumb(variant[0], t - variant[1])
        scores = {deg: corr(v, c) for deg, c in rotations(thumb(control[0], t - control[1])).items()}
        votes.append((max(scores, key=scores.get), round(scores[0], 3)))
    upright = sum(1 for deg, _ in votes if deg == 0)
    return {"ближе_всего": [deg for deg, _ in votes], "corr_0": [c for _, c in votes],
            "ok": upright >= MIN_ORIENT_VOTES * len(votes)}


def color_stats(clip: Path) -> dict:
    """Средняя насыщенность (HSV S) и контраст (σ яркости) по 5 кадрам клипа — блёклый HDR без тонмаппинга их роняет."""
    import cv2

    dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(clip)],
                                          capture_output=True, text=True, check=True).stdout)["format"]["duration"])
    sats, contrasts = [], []
    for k in range(1, 6):
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{dur * k / 6:.3f}", "-i", str(clip),
                              "-frames:v", "1", "-vf", "scale=270:480", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True, check=True).stdout
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(480, 270, 3)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[..., 1].mean()))
        contrasts.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()))
    return {"saturation": round(float(np.mean(sats)), 1), "contrast": round(float(np.mean(contrasts)), 1)}


def output_ok(clip: Path) -> tuple[bool, str]:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_transfer:stream_side_data=rotation",
         "-of", "json", str(clip)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    rotated = any("rotation" in s for s in data.get("side_data_list") or ())
    ok = (data["codec_name"] == "h264" and (data["width"], data["height"]) == (1080, 1920)
          and data["r_frame_rate"] == "30/1" and data["avg_frame_rate"] == "30/1" and data.get("pix_fmt") == "yuv420p"
          and not rotated and data.get("color_transfer") not in ("smpte2084", "arib-std-b67"))
    return ok, f'{data["codec_name"]} {data["width"]}x{data["height"]} {data["avg_frame_rate"]} {data.get("pix_fmt")} ' \
               f'trc={data.get("color_transfer")}{" ROTATED" if rotated else ""}'


def check_variant(name: str, videos: Path, runs: Path, markers: list[float], control_clips, control_colors) -> Result:
    result = Result(name)
    run_dir = runs / Path(name).stem
    run_pipeline(videos / name, run_dir)
    items = clips(run_dir)
    result.checks["клипы есть"] = bool(items)
    outputs = [output_ok(c) for c, _, _ in items]
    result.checks["выход h264 1080x1920 30 fps SDR без поворота"] = all(ok for ok, _ in outputs)
    result.details["выход"] = sorted({text for _, text in outputs})

    reference = sync.reference_onsets(videos / name)
    rows = [row for c, _, _ in items for row in sync.analyse_clip(c, markers, reference)["markers"]]
    found = [r for r in rows if r["av_offset_ms"] is not None]
    result.details["метки"] = {"в клипах": len(rows), "найдено": len(found),
                               "звук-видео мс": [min((r["av_offset_ms"] for r in found), default=None),
                                                 max((r["av_offset_ms"] for r in found), default=None)]}
    result.checks["метки найдены"] = len(found) == len(rows)
    result.checks["синхронизация ≤ 45 мс"] = all(abs(r["av_offset_ms"]) <= MAX_AV_OFFSET_MS for r in found)
    result.checks["место меток ≤ 50 мс"] = all(abs(r["video_err_ms"]) <= MAX_POSITION_ERR_MS and abs(r["audio_err_ms"]) <= MAX_POSITION_ERR_MS
                                               for r in found)

    if control_clips is not None and name != "cfr_control.mp4":
        pairs = match_moments([(s, e) for _, s, e in control_clips], [(s, e) for _, s, e in items])
        matched = [p for p in pairs if p[1] >= 0]
        result.details["моменты"] = {"контроль": [(round(s, 2), round(e, 2)) for _, s, e in control_clips],
                                     "вариант": [(round(s, 2), round(e, 2)) for _, s, e in items],
                                     "IoU": [p[2] for p in pairs]}
        result.checks["моменты совпадают с контролем"] = len(matched) >= MIN_MOMENT_MATCH * len(pairs)
        orient = [orientation(items[j], control_clips[i]) for i, j, _ in matched]
        result.details["ориентация"] = orient
        result.checks["ориентация как у контроля"] = bool(orient) and all(o["ok"] for o in orient)
    if control_colors is not None and name in maker.HDR:
        colors = [color_stats(c) for c, _, _ in items]
        sat = float(np.mean([c["saturation"] for c in colors]))
        con = float(np.mean([c["contrast"] for c in colors]))
        result.details["цвет"] = {"насыщенность": round(sat, 1), "контраст": round(con, 1),
                                  "у контроля": control_colors}
        result.checks["HDR: цвета не блёклые"] = (sat >= HDR_MIN_RATIO * control_colors["насыщенность"]
                                                  and con >= HDR_MIN_RATIO * control_colors["контраст"])
    return result


def check_real(video: Path, runs: Path) -> Result:
    result = Result(f"(настоящий) {video.name}")
    run_dir = runs / f"real_{video.stem[:40]}"
    run_pipeline(video, run_dir)
    items = clips(run_dir)
    result.checks["клипы есть"] = bool(items)
    outputs = [output_ok(c) for c, _, _ in items]
    result.checks["выход h264 1080x1920 30 fps SDR без поворота"] = all(ok for ok, _ in outputs)
    result.details["выход"] = sorted({text for _, text in outputs})
    diffs = []
    for clip, _, _ in items:
        streams = {s["codec_type"]: s for s in sync.probe(clip)["streams"]}
        diffs.append(abs(float(streams["video"].get("duration", 0)) - float(streams.get("audio", {}).get("duration", 0))))
    result.details["|видео − звук| мс"] = round(max(diffs, default=0) * 1000, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("work_dir")
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--only", nargs="*", help="варианты (имена файлов); контроль делается всегда")
    parser.add_argument("--real", nargs="*", default=[], help="настоящие ролики (без меток)")
    parser.add_argument("--json")
    args = parser.parse_args()

    work = Path(args.work_dir).expanduser()
    videos, runs = work / "videos", work / "runs"
    only = None if not args.only else sorted(set(args.only) | {"cfr_control.mp4"})
    meta = maker.generate(Path(args.source), videos, args.start, args.duration, only)
    names = [n for n in maker.variants(meta["base_fps"], meta["duration"]) if only is None or n in only]

    control = check_variant("cfr_control.mp4", videos, runs, meta["markers"], None, None)
    control_clips = clips(runs / "cfr_control")
    colors = [color_stats(c) for c, _, _ in control_clips]
    control_colors = {"насыщенность": round(float(np.mean([c["saturation"] for c in colors])), 1),
                      "контраст": round(float(np.mean([c["contrast"] for c in colors])), 1)} if colors else None
    results = [control] + [check_variant(n, videos, runs, meta["markers"], control_clips, control_colors)
                           for n in names if n != "cfr_control.mp4"]
    results += [check_real(Path(v).expanduser(), runs) for v in args.real]

    failed = False
    for res in results:
        failed |= not res.ok
        print(f"{'OK  ' if res.ok else 'FAIL'} {res.name}")
        for check, ok in res.checks.items():
            if not ok:
                print(f"       не прошло: {check}")
        for key, value in res.details.items():
            print(f"       {key}: {value}")
    if args.json:
        Path(args.json).write_text(json.dumps([{"name": r.name, "ok": r.ok, "checks": r.checks, "details": r.details}
                                               for r in results], ensure_ascii=False, indent=2, default=str))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
