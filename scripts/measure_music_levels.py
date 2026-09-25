"""Замеры громкости музыки в готовых клипах: старый фиксированный множитель 0.2 против смещения в дБ от оригинала.

    python scripts/measure_music_levels.py OUT_DIR PARTS_DIR [--offset -14] клип1 клип2 ...

OUT_DIR — папка с клипами и их JSON (start_sec, музыка), PARTS_DIR — исходные видео (из JSON source_video).
Для каждого клипа: громкость оригинального звука, трека, музыки «до» (трек × 0.2) и «после» (усиление по offset),
разница музыка−оригинал в дБ и громкость всего клипа. «После» — настоящий mix_music на копии клипа с оригинальным звуком.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.music_mixer import measure_lufs, mix_music, plan_gain, probe_duration, track_lufs  # noqa: E402

MUSIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "music"


def stem_lufs(track: Path, gain_db: float, duration: float, tmp: Path) -> float | None:
    """Громкость музыкального «стема»: трек с усилением gain_db (зациклен), первые duration секунд."""
    wav = tmp / "stem.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-stream_loop", "-1", "-i", str(track), "-t", f"{duration:.3f}",
         "-af", f"volume={gain_db:.3f}dB", str(wav)], check=True,
    )
    return measure_lufs(wav)


def main() -> None:
    args = sys.argv[1:]
    offset = -14.0
    if "--offset" in args:
        i = args.index("--offset"); offset = float(args[i + 1]); del args[i:i + 2]
    out_dir, parts_dir, names = Path(args[0]), Path(args[1]), args[2:]
    header = f"{'клип':16s} {'оригинал':>9s} | {'трек':>6s} | ДО: музыка  Δ к ориг.  клип целиком | ПОСЛЕ: усил.  музыка  Δ к ориг.  клип целиком"
    print(header)
    for name in names:
        meta = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
        track = MUSIC_DIR / meta["music_file"]
        start, duration = meta["start_sec"], probe_duration(out_dir / f"{name}.mp4")
        source = parts_dir / meta["source_video"]
        original = measure_lufs(source, start, duration)
        track_l = track_lufs(track, duration)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            old_stem = stem_lufs(track, 20 * __import__("math").log10(0.2), duration, tmp)
            old_total = measure_lufs(out_dir / f"{name}.mp4")          # клип, собранный старой версией
            clean = tmp / "clean.mp4"                                    # то же видео, но с оригинальным звуком без музыки
            subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(out_dir / f"{name}.mp4"), "-ss", f"{start:.3f}",
                            "-t", f"{duration:.3f}", "-i", str(source), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(clean)], check=True)
            report = mix_music(clean, track, duration, offset, [], -12.0)
            gain, _ = plan_gain(original, track_l, offset)
            new_stem = stem_lufs(track, gain, duration, tmp)
            new_total = measure_lufs(clean)
        d_old = old_stem - original
        d_new = new_stem - original
        print(f"{name:16s} {original:8.1f}  | {track_l:6.1f} | {old_stem:8.1f}   {d_old:+7.1f} дБ   {old_total:8.1f}     | {gain:+7.1f}  {new_stem:8.1f}   {d_new:+7.1f} дБ   {new_total:8.1f}")


if __name__ == "__main__":
    main()
