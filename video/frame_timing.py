"""Видео с переменной частотой кадров (VFR, телефоны): анализ по CFR-копии.

Анализ читает кадры OpenCV подряд и считает время кадра как «номер / fps» (video/frame_extractor.py,
video/shared_decode.py, PySceneDetect). OpenCV отдаёт средний fps, поэтому у VFR номер кадра расходится со
временем: телефон в темноте снимает минуту на 15 fps вместо 30 — и к концу этой минуты «номер / 25» опережает
реальное время на 12 с. Кроп-трекинг тогда следит за животным из другого места ролика, окна и сцены смещены,
а экспорт (ffmpeg, -ss по реальному времени) режет правильный кусок — кроп и картинка расходятся.

Проверка дешёвая: ffprobe читает только метки времени пакетов (без декодирования) и считает наибольшее
расхождение реального времени кадра с «номер / средний fps». Больше MAX_INDEX_DRIFT_SEC — анализ (сцены, окна,
прыжки, кроп, обложка) идёт по CFR-копии: ffmpeg дублирует/выбрасывает кадры по реальному времени, так что
«номер / fps» копии — это реальное время оригинала. Звук, транскрипция и экспорт берутся из оригинала
(ffmpeg режет VFR по времени, а в экспорте fps=30 стоит первым фильтром). Разрешение копии то же, что у
оригинала: координаты кропа из анализа применяются к оригиналу.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

MAX_INDEX_DRIFT_SEC = 0.1     # больше ~3 кадров при 30 fps: кроп/сцены заметно съезжают
PROXY_MAX_FPS = 60.0
PROXY_MIN_FPS = 24.0


@dataclass(frozen=True, slots=True)
class FrameTiming:
    frames: int
    avg_fps: float                # как видит OpenCV (CAP_PROP_FPS = avg_frame_rate)
    nominal_fps: float            # r_frame_rate: наибольшая «базовая» частота потока
    max_drift_sec: float          # max |реальное время кадра − номер / avg_fps|
    min_interval_ms: float
    max_interval_ms: float

    @property
    def needs_cfr_proxy(self) -> bool:
        return self.max_drift_sec > MAX_INDEX_DRIFT_SEC

    @property
    def proxy_fps(self) -> float:
        return min(PROXY_MAX_FPS, max(PROXY_MIN_FPS, round(self.nominal_fps or self.avg_fps or 30.0)))


def _rate(raw: str | None) -> float:
    try:
        num, den = (raw or "0/1").split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_frame_timing(path: Path, timeout: float = 300.0) -> FrameTiming | None:
    """Метки времени всех кадров видеодорожки (без декодирования). None — ffprobe не справился."""
    try:
        stream = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=timeout,
        ).stdout)["streams"][0]
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, check=True, timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        logger.warning("Метки времени кадров {} не прочитаны: {} — считаю частоту кадров постоянной", path.name, exc)
        return None
    times = sorted(float(x) for x in raw.split() if x.strip() and x.strip() != "N/A")
    avg_fps = _rate(stream.get("avg_frame_rate"))
    if len(times) < 2 or avg_fps <= 0:
        return None
    first = times[0]
    drift = max(abs((t - first) - k / avg_fps) for k, t in enumerate(times))
    intervals = [b - a for a, b in zip(times, times[1:])]
    return FrameTiming(
        frames=len(times), avg_fps=avg_fps, nominal_fps=_rate(stream.get("r_frame_rate")), max_drift_sec=drift,
        min_interval_ms=min(intervals) * 1000, max_interval_ms=max(intervals) * 1000,
    )


def make_cfr_proxy(path: Path, output: Path, fps: float, timeout: float = 3600.0) -> Path:
    """CFR-копия видеодорожки для анализа (без звука). Быстрое кодирование: копия живёт до конца обработки видео."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", f"fps={fps:g}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-g", str(int(fps)),
        str(output),
    ]
    logger.info("CFR-копия для анализа: {}", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    return output
