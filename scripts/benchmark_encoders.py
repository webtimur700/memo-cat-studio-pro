"""Сравнение кодеров libx264 и h264_vaapi: время, размер, качество (VMAF/PSNR) и ошибки декодирования.

    python scripts/benchmark_encoders.py video.mp4 [--start 30] [--seconds 15]

Кадр приводится к тому же виду, что в пайплайне (кроп 9:16 из центра, scale 1080x1920, 30 fps); эталон —
то же самое без потерь (FFV1). Кодеры и настройки берутся из export/encoder.py, чтобы замер не расходился с
реальным экспортом.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.entities.settings import ExportSettings  # noqa: E402
from export.encoder import build_encoder_args, vaapi_h264_works  # noqa: E402
from export.quality_presets import resolve_quality_preset  # noqa: E402

BASE_FILTER = "crop=ih*9/16:ih,scale=1080:1920,fps=30"


def run(command: list[str]) -> tuple[float, str]:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    return time.perf_counter() - started, result.stderr


def quality(encoded: Path, reference: Path) -> tuple[str, str]:
    align = "setpts=N/30/TB"
    _, err = run(["ffmpeg", "-hide_banner", "-i", str(encoded), "-i", str(reference), "-lavfi",
                  f"[0:v]{align}[a];[1:v]{align}[b];[a][b]libvmaf=n_threads=8", "-f", "null", "-"])
    vmaf = re.search(r"VMAF score: ([\d.]+)", err)
    _, err = run(["ffmpeg", "-hide_banner", "-i", str(encoded), "-i", str(reference), "-lavfi",
                  f"[0:v]{align}[a];[1:v]{align}[b];[a][b]psnr", "-f", "null", "-"])
    psnr = re.search(r"average:([\d.]+)", err)
    return (vmaf.group(1) if vmaf else "?"), (psnr.group(1) if psnr else "?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=30)
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    source = ["-ss", str(args.start), "-t", str(args.seconds), "-i", str(Path(args.video).expanduser())]

    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "ref.mkv"
        run(["ffmpeg", "-y", "-loglevel", "error", *source, "-vf", BASE_FILTER, "-c:v", "ffv1", "-an", str(reference)])
        print(f"VAAPI работает: {vaapi_h264_works()}")
        header = f"{'кодер':<14}{'пресет':<8}{'время, с':>9}{'размер, МБ':>11}{'Мбит/с':>8}{'VMAF':>7}{'PSNR':>7}{'ошибок декод.':>14}"
        print(header)

        def report(label: str, preset: str, out: Path, elapsed: float) -> None:
            _, decode_err = run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"])
            vmaf, psnr = quality(out, reference)
            size = out.stat().st_size
            print(f"{label:<14}{preset:<8}{elapsed:>9.2f}{size / 1e6:>11.2f}{size * 8 / args.seconds / 1e6:>8.2f}"
                  f"{float(vmaf):>7.2f}{float(psnr):>7.2f}{len(decode_err.splitlines()):>14}")

        for preset in ("high", "medium", "low"):
            settings = ExportSettings(quality_preset=preset)
            quality_preset = resolve_quality_preset(preset)
            enc = build_encoder_args("x264", settings, quality_preset)
            out = Path(tmp) / f"x264_{preset}.mp4"
            elapsed, _ = run(["ffmpeg", "-y", "-loglevel", "error", *source, "-vf", BASE_FILTER, *enc.codec_args, "-an", str(out)])
            report("libx264", preset, out, elapsed)
            x264_mbps = out.stat().st_size * 8 / args.seconds / 1e6
            if not vaapi_h264_works():
                continue
            enc = build_encoder_args("vaapi", settings, quality_preset)
            out = Path(tmp) / f"vaapi_{preset}.mp4"
            vf = ",".join((BASE_FILTER, enc.filter_suffix))
            elapsed, _ = run(["ffmpeg", "-y", "-loglevel", "error", *enc.global_args, *source, "-vf", vf, *enc.codec_args, "-an", str(out)])
            report("vaapi (по доле)", preset, out, elapsed)
            # тот же размер файла, что дал libx264: честное сравнение качества
            matched = [c for c in enc.codec_args]
            matched[matched.index("-b:v") + 1] = f"{x264_mbps:.2f}M"
            matched[matched.index("-maxrate") + 1] = f"{2 * x264_mbps:.2f}M"
            matched[matched.index("-bufsize") + 1] = f"{4 * x264_mbps:.2f}M"
            out = Path(tmp) / f"vaapi_matched_{preset}.mp4"
            elapsed, _ = run(["ffmpeg", "-y", "-loglevel", "error", *enc.global_args, *source, "-vf", vf, *matched, "-an", str(out)])
            report("vaapi (=размер)", preset, out, elapsed)


if __name__ == "__main__":
    main()
