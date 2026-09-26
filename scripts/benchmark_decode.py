"""Скорость декодирования видео разными способами (кадров/с) на куске видео.

    python scripts/benchmark_decode.py video.mp4 [--start 600] [--seconds 120]

Способы: OpenCV (grab / read / read + уменьшение), ffmpeg программно и через VAAPI, с уменьшением кадра на выходе
декодера (до размера кадра детектора сцен и до входа YOLO). Всё — до /dev/null, без анализа кадров: только цена
декодирования и конвертации. Запускать при свободном CPU, иначе цифры занижены.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2

VAAPI_DEVICE = "/dev/dri/renderD128"
SCENE_W, SCENE_H = 274, 154      # 1920x1080 / 7: кадр детектора сцен (scenedetect auto_downscale)
YOLO_W, YOLO_H = 640, 360        # вход YOLO 640x640 с сохранением пропорций (остальное — серая рамка)


def ffmpeg_variant(video: Path, start: float, seconds: float, hw: bool, vf: str, out_fmt: str | None) -> float:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(seconds)]
    if hw:
        cmd += ["-hwaccel", "vaapi", "-hwaccel_device", VAAPI_DEVICE, "-hwaccel_output_format", "vaapi"]
    cmd += ["-i", str(video), "-an", "-sn"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", out_fmt, "-"] if out_fmt else ["-f", "null", "-"]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"код {result.returncode}")
    return time.perf_counter() - t0


def cv_variant(video: Path, start: float, seconds: float, mode: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = int(seconds * fps)
    t0 = time.perf_counter()
    for _ in range(frames):
        if mode == "grab":
            ok = cap.grab()
        else:
            ok, frame = cap.read()
            if ok and mode == "read_resize":
                cv2.resize(frame, (SCENE_W, SCENE_H), interpolation=cv2.INTER_LINEAR)
        if not ok:
            break
    return time.perf_counter() - t0, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args()
    video = Path(args.video).expanduser()

    probe = cv2.VideoCapture(str(video))
    frames = int(args.seconds * probe.get(cv2.CAP_PROP_FPS))
    probe.release()

    rows: list[tuple[str, float, int]] = []
    for mode in ("grab", "read", "read_resize"):
        sec, n = cv_variant(video, args.start, args.seconds, mode)
        rows.append((f"OpenCV {mode}", sec, n))
    hw_upload = "hwdownload,format=nv12"
    variants = [
        ("ffmpeg CPU: декодирование", False, "", None),
        ("ffmpeg CPU: декод. + уменьш. до кадра сцен (bgr24)", False, f"scale={SCENE_W}:{SCENE_H}", "bgr24"),
        ("ffmpeg CPU: декод. + уменьш. до входа YOLO (bgr24)", False, f"scale={YOLO_W}:{YOLO_H}", "bgr24"),
        ("ffmpeg VAAPI: декодирование", True, "", None),
        ("ffmpeg VAAPI: декод. + уменьш. до кадра сцен (bgr24)", True, f"scale_vaapi=w={SCENE_W}:h={SCENE_H}:format=nv12,{hw_upload}", "bgr24"),
        ("ffmpeg VAAPI: декод. + уменьш. до входа YOLO (bgr24)", True, f"scale_vaapi=w={YOLO_W}:h={YOLO_H}:format=nv12,{hw_upload}", "bgr24"),
        ("ffmpeg VAAPI: декод., кадр 1920x1080 на CPU (bgr24)", True, f"scale_vaapi=format=nv12,{hw_upload}", "bgr24"),
    ]
    for name, hw, vf, fmt in variants:
        try:
            rows.append((name, ffmpeg_variant(video, args.start, args.seconds, hw, vf, fmt), frames))
        except RuntimeError as exc:
            print(f"{name}: не работает ({exc})")

    print(f"\nкусок {args.seconds:.0f} с с {args.start:.0f}-й секунды, {frames} кадров\n")
    print(f"{'способ':62s} {'с':>7s} {'кадров/с':>9s}")
    for name, sec, n in rows:
        print(f"{name:62s} {sec:7.1f} {n / sec:9.0f}")


if __name__ == "__main__":
    main()
