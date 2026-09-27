"""Тестовые видео «не как myvideo.mp4»: переменная частота кадров, 4K, высокий битрейт.

Все варианты — один и тот же отрезок исходника с метками синхронизации: в известные моменты весь кадр
белеет на 100 мс и одновременно звучит писк 3 кГц на 100 мс. По меткам в готовых клипах
scripts/check_av_sync.py меряет рассинхрон звука и видео и точность границ момента (метка должна
оказаться в клипе ровно там, где ей велит начало момента из JSON), не зная, как клип кадрирован.

    python scripts/make_test_videos.py ~/Видео/myvideo.mp4 OUT_DIR [--start 120 --duration 180]

Создаёт в OUT_DIR:
  cfr1080p60.mp4      — контроль: 1080p60 H.264, как исходник
  vfr1080.mp4         — переменная частота: ~40% кадров выброшено случайно, метки времени кадров сохранены
                        (так пишут телефоны: интервалы между кадрами неравные, звук непрерывный)
  uhd4k30_hevc.mp4    — 3840x2160, 30 fps, HEVC 45 Мбит/с (как 4K с телефона)
  uhd4k60_h264.mp4    — 3840x2160, 60 fps, H.264 100 Мбит/с (самое тяжёлое декодирование)
  hibitrate1080p60.mp4 — 1080p60 H.264 80 Мбит/с
  markers.json        — времена меток (секунды от начала тестового видео)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FLASH_SEC = 0.1
TONE_HZ = 3000


def marker_times(duration: float) -> list[float]:
    """Неравные интервалы 9–15 с: метка не должна совпадать с сеткой кадров/окон пайплайна."""
    times, t, k = [], 3.0, 0
    while t < duration - 2:
        times.append(round(t, 3))
        k += 1
        t += 9 + (k * 3.7) % 6
    return times


def filters(times: list[float], duration: float) -> tuple[str, str]:
    enable = "+".join(f"between(t,{t},{t + FLASH_SEC})" for t in times)
    video = f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{enable}'"
    tone = f"aevalsrc='0.6*sin(2*PI*{TONE_HZ}*t)*({enable})':s=44100:d={duration}"
    return video, tone


VARIANTS = {
    "cfr1080p60.mp4": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"], None),
    "vfr1080.mp4": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-fps_mode", "vfr"], "select='gt(random(0),0.4)'"),
    "uhd4k30_hevc.mp4": (
        ["-c:v", "libx265", "-preset", "ultrafast", "-b:v", "45M", "-tag:v", "hvc1", "-x265-params", "log-level=error"],
        "fps=30,scale=3840:2160:flags=lanczos",
    ),
    "uhd4k60_h264.mp4": (
        ["-c:v", "libx264", "-preset", "ultrafast", "-b:v", "100M", "-maxrate", "120M", "-bufsize", "120M"],
        "scale=3840:2160:flags=lanczos",
    ),
    "hibitrate1080p60.mp4": (
        ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "80M", "-maxrate", "100M", "-bufsize", "100M"], None,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("out_dir")
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--only", nargs="*", help="какие варианты делать (имена файлов)")
    args = parser.parse_args()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    times = marker_times(args.duration)
    (out / "markers.json").write_text(json.dumps(
        {"source": str(Path(args.source).expanduser()), "start": args.start, "duration": args.duration,
         "flash_sec": FLASH_SEC, "tone_hz": TONE_HZ, "markers": times}, indent=2))
    mark_video, tone = filters(times, args.duration)

    for name, (codec, extra) in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        chain = mark_video + (f",{extra}" if extra else "")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(args.start), "-t", str(args.duration), "-i", str(Path(args.source).expanduser()),
            "-f", "lavfi", "-t", str(args.duration), "-i", tone,
            "-filter_complex", f"[0:v]{chain},format=yuv420p[v];[0:a][1:a]amix=inputs=2:normalize=0:duration=first[a]",
            "-map", "[v]", "-map", "[a]", *codec, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(out / name),
        ]
        print(name, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
