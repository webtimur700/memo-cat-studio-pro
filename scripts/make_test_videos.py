"""Набор тестовых видео «не как myvideo.mp4» из любого исходника: VFR, 4K, высокий битрейт, поворот, HDR.

Все варианты — один и тот же отрезок исходника с метками синхронизации: в известные моменты весь кадр
белеет на 100 мс и одновременно звучит писк 3 кГц на 100 мс. По меткам в готовых клипах
scripts/check_av_sync.py меряет рассинхрон звука и видео и точность границ момента, не зная, как клип
кадрирован; scripts/regress_video_variants.py прогоняет весь набор и сверяет с контрольной CFR-копией.

    python scripts/make_test_videos.py SOURCE OUT_DIR [--start 120 --duration 180] [--only имя.mp4 ...]

Базовая частота R = частота исходника, но не больше 60 (всё приводится к CFR R). Создаёт в OUT_DIR:
  cfr_control.mp4     — контроль: разрешение исходника, CFR R fps, H.264 CRF 18
  vfr_random.mp4      — VFR: ~40% кадров выброшено случайно, метки времени кадров сохранены
  vfr_lowlight.mp4    — VFR как телефон в темноте: R/2 fps, в средней трети ролика R/4 fps
                        (номер кадра / средний fps расходится со временем на секунды)
  uhd4k30_hevc.mp4    — 4K по длинной стороне (3840), 30 fps, HEVC 45 Мбит/с (как 4K с телефона)
  uhd4k60_h264.mp4    — 4K по длинной стороне, R fps, H.264 100 Мбит/с (самое тяжёлое декодирование)
  hibitrate.mp4       — разрешение исходника, R fps, H.264 80 Мбит/с
  rot_m90.mp4, rot_p90.mp4, rot180.mp4
                      — как телефон: кадры сохранены повёрнутыми, правильную ориентацию задаёт тег поворота
                        (display matrix −90°, +90°, 180°); при показе должны совпадать с cfr_control
  hdr_hlg.mp4, hdr_pq.mp4
                      — 10 бит HEVC Main10, BT.2020, HLG (как iPhone) и PQ (HDR10): SDR-исходник переведён в HDR
                        (белый SDR = 203 нит), так что правильный тонмаппинг возвращает исходные цвета
  markers.json        — времена меток (секунды от начала тестового видео) и параметры набора
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FLASH_SEC = 0.1
TONE_HZ = 3000
UHD_LONG_SIDE = 3840
MAX_BASE_FPS = 60

# цветовые метаданные HDR-вариантов: (transfer для zscale, transfer для x265/ffmpeg)
HDR = {"hdr_hlg.mp4": ("arib-std-b67", "arib-std-b67"), "hdr_pq.mp4": ("smpte2084", "smpte2084")}
ROTATIONS = {"rot_m90.mp4": (-90, "transpose=2"), "rot_p90.mp4": (90, "transpose=1"), "rot180.mp4": (180, "hflip,vflip")}


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


def probe(source: Path) -> dict:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,r_frame_rate,width,height:format=duration",
         "-of", "json", str(source)], capture_output=True, text=True, check=True).stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = video["r_frame_rate"].split("/")
    return {
        "fps": float(num) / float(den), "width": int(video["width"]), "height": int(video["height"]),
        "duration": float(data["format"]["duration"]), "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def uhd_scale() -> str:
    return (f"scale='if(gte(iw,ih),{UHD_LONG_SIDE},-2)':'if(gte(iw,ih),-2,{UHD_LONG_SIDE})':flags=lanczos")


def variants(base_fps: int, duration: float) -> dict[str, tuple[list[str], str | None]]:
    """имя -> (параметры кодера, доп. фильтры после меток и приведения к CFR base_fps)."""
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    lo, hi = round(duration / 3, 3), round(2 * duration / 3, 3)
    result: dict[str, tuple[list[str], str | None]] = {
        "cfr_control.mp4": (x264, None),
        "vfr_random.mp4": ([*x264, "-fps_mode", "vfr"], "select='gt(random(0),0.4)'"),
        "vfr_lowlight.mp4": ([*x264, "-fps_mode", "vfr"], f"select='if(between(t,{lo},{hi}),not(mod(n,4)),not(mod(n,2)))'"),
        "uhd4k30_hevc.mp4": (
            ["-c:v", "libx265", "-preset", "ultrafast", "-b:v", "45M", "-tag:v", "hvc1", "-x265-params", "log-level=error"],
            f"fps=30,{uhd_scale()}",
        ),
        "uhd4k60_h264.mp4": (
            ["-c:v", "libx264", "-preset", "ultrafast", "-b:v", "100M", "-maxrate", "120M", "-bufsize", "120M"], uhd_scale(),
        ),
        "hibitrate.mp4": (["-c:v", "libx264", "-preset", "veryfast", "-b:v", "80M", "-maxrate", "100M", "-bufsize", "100M"], None),
    }
    for name, (_, stored) in ROTATIONS.items():
        result[name] = (x264, stored)
    for name, (zs_transfer, x265_transfer) in HDR.items():
        # SDR BT.709 -> линейный свет (белый SDR = 203 нит, как в BT.2408) -> BT.2020 + HLG/PQ, 10 бит
        to_hdr = (f"zscale=tin=bt709:min=bt709:pin=bt709:rin=tv:t=linear:npl=203,format=gbrpf32le,"
                  f"zscale=p=bt2020:t={zs_transfer}:m=bt2020nc:r=tv:npl=203,format=yuv420p10le")
        result[name] = ([
            "-c:v", "libx265", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
            "-color_primaries", "bt2020", "-color_trc", x265_transfer, "-colorspace", "bt2020nc", "-color_range", "tv",
            "-x265-params", f"log-level=error:colorprim=bt2020:transfer={x265_transfer}:colormatrix=bt2020nc:range=limited",
        ], to_hdr)
    return result


def generate(source: Path, out_dir: Path, start: float = 120.0, duration: float = 180.0,
             only: list[str] | None = None) -> dict:
    """Делает варианты (уже существующие пропускает) и markers.json; возвращает содержимое markers.json."""
    source = source.expanduser()
    info = probe(source)
    start = min(start, max(0.0, info["duration"] - 20))
    duration = min(duration, info["duration"] - start)
    base_fps = int(min(MAX_BASE_FPS, round(info["fps"]) or 30))
    out_dir.mkdir(parents=True, exist_ok=True)
    times = marker_times(duration)
    meta = {"source": str(source), "start": start, "duration": duration, "base_fps": base_fps, "flash_sec": FLASH_SEC,
            "tone_hz": TONE_HZ, "markers": times, "control": "cfr_control.mp4",
            "rotations": {name: deg for name, (deg, _) in ROTATIONS.items()}, "hdr": sorted(HDR)}
    (out_dir / "markers.json").write_text(json.dumps(meta, indent=2))
    mark_video, tone = filters(times, duration)
    audio_in = ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=44100:cl=stereo"] if not info["has_audio"] else []
    audio_label = "[2:a]" if not info["has_audio"] else "[0:a]"

    for name, (codec, extra) in variants(base_fps, duration).items():
        target = out_dir / name
        if (only and name not in only) or target.exists():
            continue
        chain = f"{mark_video},fps={base_fps}" + (f",{extra}" if extra else "")
        rotation = ROTATIONS.get(name)
        encode_to = Path(tempfile.mkdtemp(dir=out_dir)) / name if rotation else target
        pix_fmt = "" if name in HDR else ",format=yuv420p"   # HDR: 10 бит задаёт свой фильтр
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(duration), "-i", str(source),
            "-f", "lavfi", "-t", str(duration), "-i", tone, *audio_in,
            "-filter_complex", f"[0:v]{chain}{pix_fmt}[v];{audio_label}[1:a]amix=inputs=2:normalize=0:duration=first[a]",
            "-map", "[v]", "-map", "[a]", *codec, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(encode_to),
        ]
        print(name, flush=True)
        subprocess.run(cmd, check=True)
        if rotation:
            # тег поворота: плеер (и ffmpeg при экспорте) поворачивает сохранённый кадр обратно в исходную ориентацию
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", str(rotation[0]),
                            "-i", str(encode_to), "-c", "copy", "-movflags", "+faststart", str(target)], check=True)
            encode_to.unlink()
            encode_to.parent.rmdir()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("out_dir")
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--only", nargs="*", help="какие варианты делать (имена файлов)")
    args = parser.parse_args()
    generate(Path(args.source), Path(args.out_dir).expanduser(), args.start, args.duration, args.only)


if __name__ == "__main__":
    main()
