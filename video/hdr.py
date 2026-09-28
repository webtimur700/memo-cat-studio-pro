"""HDR-исходники (iPhone: HEVC 10 бит, BT.2020, HLG; HDR10: PQ) -> SDR BT.709 для клипа, обложки и кадра LLM.

Без преобразования ffmpeg и OpenCV просто урезают 10 бит до 8 и выдают значения HLG/PQ как будто это BT.709:
картинка блёклая и серая (PQ — особенно), а клип при этом помечен как обычный SDR. Здесь — один и тот же
тонмаппинг для экспорта (фильтр в начале цепочки) и для отдельных кадров (обложка и кадр для LLM).

Цепочка: zscale в линейный свет (npl — яркость белого SDR, 203 нит по BT.2408) -> float RGB -> BT.709 праймериз ->
tonemap (mobius, param=0.9: линейно до 90% белого SDR, выше мягко сжимает света; при параметре по умолчанию 0.3 сжималось
всё ярче 30% белого — белый выходил 216 вместо 255; hable темнит SDR-часть) -> гамма BT.709, ограниченный диапазон, 8 бит.
Замер на SDR-кадре, переведённом в HLG/PQ и обратно: отклонение 1.1–1.3 из 255 (живой кадр), 9.6 на таблице testsrc2 при
погрешности самого перевода BT.709 -> BT.2020 10 бит -> BT.709 8.5 (docs/video_variants.md).
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
SDR_WHITE_NITS = 203

TONEMAP_CHAIN = (
    f"zscale=t=linear:npl={SDR_WHITE_NITS},format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=mobius:param=0.9:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


@lru_cache(maxsize=64)
def hdr_transfer(path: Path) -> str | None:
    """'smpte2084' (PQ) / 'arib-std-b67' (HLG) для HDR-видео, None для SDR или если ffprobe не справился."""
    try:
        stream = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_transfer", "-of", "json", str(path)],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout)["streams"][0]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        logger.warning("Цветовые метаданные {} не прочитаны: {} — считаю видео SDR", path.name, exc)
        return None
    transfer = stream.get("color_transfer")
    return transfer if transfer in HDR_TRANSFERS else None


def tonemapped_frame(path: Path, timestamp_sec: float, width: int, height: int) -> np.ndarray:
    """Кадр BGR (как у OpenCV, в видимой ориентации) из HDR-видео, переведённый в SDR той же цепочкой, что и клип."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, timestamp_sec):.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"{TONEMAP_CHAIN},format=bgr24", "-f", "rawvideo", "-"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    expected = width * height * 3
    if len(raw) < expected:
        raise ValueError(f"кадр {timestamp_sec:.2f} с из {path.name}: {len(raw)} байт вместо {expected}")
    return np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3).copy()
