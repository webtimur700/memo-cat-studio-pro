"""Проверка готовых клипов по меткам scripts/make_test_videos.py: синхронизация и границы момента.

Для каждого клипа (*_momentN.mp4 + .json рядом):
  * ffprobe: кодек, размер, fps, длительности видео и звука, начала потоков;
  * вспышки (средняя яркость кадра) и писки (энергия в полосе 2.7–3.3 кГц) — моменты в клипе;
  * рассинхрон = писк − вспышка для каждой метки, попавшей в клип (>0 — звук опаздывает);
  * граница: где метка должна быть по start_sec из JSON (метка − start_sec) и где она в клипе.
    Отклонение видео/звука от ожидаемого — ошибка позиционирования (seek, ts исходника, VFR).

    python scripts/check_av_sync.py CLIPS_DIR MARKERS_JSON [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

AUDIO_RATE = 48000
WINDOW_SEC = 0.002
LUMA_W, LUMA_H = 36, 64


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration,start_time,nb_frames,sample_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def frame_luma(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(время кадра по pts, средняя яркость) — pts берутся из showinfo, а не из номера кадра."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(path), "-map", "0:v:0",
         "-vf", f"scale={LUMA_W}:{LUMA_H},format=gray,showinfo", "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, LUMA_H * LUMA_W)
    times = [float(line.split("pts_time:")[1].split()[0]) for line in proc.stderr.decode("utf-8", "replace").splitlines()
             if "pts_time:" in line]
    return np.asarray(times[: len(frames)]), frames.mean(axis=1)


def tone_envelope(path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
         "-af", "highpass=f=2700,highpass=f=2700,lowpass=f=3300,lowpass=f=3300", "-ac", "1", "-ar", str(AUDIO_RATE),
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    win = int(AUDIO_RATE * WINDOW_SEC)
    n = len(samples) // win
    return np.sqrt((samples[: n * win].reshape(n, win) ** 2).mean(axis=1))


def onsets(times: np.ndarray, values: np.ndarray, threshold: float, min_gap: float = 1.0) -> list[float]:
    result: list[float] = []
    above = values > threshold
    for i in range(len(values)):
        if above[i] and (i == 0 or not above[i - 1]) and (not result or times[i] - result[-1] > min_gap):
            result.append(float(times[i]))
    return result


def analyse_clip(clip: Path, markers: list[float]) -> dict:
    meta = json.loads(clip.with_suffix(".json").read_text())
    start, end = meta["start_sec"], meta["end_sec"]
    info = probe(clip)
    v_times, luma = frame_luma(clip)
    median = float(np.median(luma))
    flashes = onsets(v_times, luma, median + 0.5 * (255 - median))
    env = tone_envelope(clip)
    a_times = np.arange(len(env)) * WINDOW_SEC
    peak = float(env.max()) if len(env) else 0.0
    beeps = onsets(a_times, env, 0.35 * peak) if peak > 500 else []

    expected = [m - start for m in markers if start + 0.05 <= m <= end - 0.15]
    rows = []
    for exp in expected:
        flash = min(flashes, key=lambda f: abs(f - exp), default=None)
        beep = min(beeps, key=lambda b: abs(b - exp), default=None)
        flash = flash if flash is not None and abs(flash - exp) < 0.5 else None
        beep = beep if beep is not None and abs(beep - exp) < 0.5 else None
        rows.append({
            "expected_sec": round(exp, 3),
            "video_err_ms": None if flash is None else round((flash - exp) * 1000, 1),
            "audio_err_ms": None if beep is None else round((beep - exp) * 1000, 1),
            "av_offset_ms": None if flash is None or beep is None else round((beep - flash) * 1000, 1),
        })
    streams = {s["codec_type"]: s for s in info["streams"]}
    v, a = streams.get("video", {}), streams.get("audio", {})
    return {
        "clip": clip.name, "start_sec": start, "end_sec": end, "moment_len": round(end - start, 3),
        "video": f'{v.get("codec_name")} {v.get("width")}x{v.get("height")} r={v.get("r_frame_rate")} avg={v.get("avg_frame_rate")}',
        "v_duration": float(v.get("duration", 0)), "a_duration": float(a.get("duration", 0)),
        "v_start": float(v.get("start_time", 0)), "a_start": float(a.get("start_time", 0)),
        "frames": int(v.get("nb_frames", 0)),
        "pts_gaps_max_ms": round(float(np.diff(v_times).max()) * 1000, 1) if len(v_times) > 1 else None,
        "unmatched_flashes": len(flashes) - sum(r["video_err_ms"] is not None for r in rows),
        "markers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips_dir")
    parser.add_argument("markers_json")
    parser.add_argument("--json")
    args = parser.parse_args()
    markers = json.loads(Path(args.markers_json).read_text())["markers"]
    results = [analyse_clip(c, markers) for c in sorted(Path(args.clips_dir).glob("*_moment*.mp4"))]

    all_rows = [r for res in results for r in res["markers"]]
    for res in results:
        print(f'{res["clip"]}: {res["video"]}, момент {res["start_sec"]:.2f}–{res["end_sec"]:.2f} ({res["moment_len"]:.2f} с), '
              f'видео {res["v_duration"]:.3f} с / звук {res["a_duration"]:.3f} с, начала {res["v_start"]}/{res["a_start"]}, '
              f'макс. интервал кадров {res["pts_gaps_max_ms"]} мс')
        for r in res["markers"]:
            print(f'    метка в {r["expected_sec"]:7.3f} с: видео {r["video_err_ms"]} мс, звук {r["audio_err_ms"]} мс, '
                  f'звук−видео {r["av_offset_ms"]} мс')
    offsets = [r["av_offset_ms"] for r in all_rows if r["av_offset_ms"] is not None]
    v_errs = [r["video_err_ms"] for r in all_rows if r["video_err_ms"] is not None]
    a_errs = [r["audio_err_ms"] for r in all_rows if r["audio_err_ms"] is not None]
    missed = sum(r["video_err_ms"] is None or r["audio_err_ms"] is None for r in all_rows)
    summary = {
        "clips": len(results), "markers_in_clips": len(all_rows), "missed": missed,
        "av_offset_ms": [min(offsets, default=None), max(offsets, default=None)],
        "video_err_ms": [min(v_errs, default=None), max(v_errs, default=None)],
        "audio_err_ms": [min(a_errs, default=None), max(a_errs, default=None)],
        "max_duration_mismatch_ms": round(max((abs(r["v_duration"] - r["moment_len"]) for r in results), default=0) * 1000, 1),
        "max_av_duration_diff_ms": round(max((abs(r["v_duration"] - r["a_duration"]) for r in results), default=0) * 1000, 1),
    }
    print("\nИТОГО:", json.dumps(summary, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "clips": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
