"""VFR и повёрнутые (телефонные) видео: кадры анализа должны соответствовать реальному времени и видимой ориентации."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from video.ffmpeg_wrapper import FFmpegWrapper
from video.frame_extractor import FrameExtractor
from video.frame_timing import make_cfr_proxy, probe_frame_timing

FLASH_AT = 5.0


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _flash_video(path: Path, vfr: bool) -> Path:
    """8 с, 30 fps; белая вспышка ровно в FLASH_AT. vfr=True: с 1-й по 4-ю секунду 10 fps (как телефон в темноте) —
    номер кадра / средний fps тогда показывает вспышку заметно позже 5 с."""
    chain = f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,{FLASH_AT},{FLASH_AT + 0.2})'"
    if vfr:
        chain += ",select='if(between(t,1,4),not(mod(n,3)),1)'"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x303030:s=320x180:r=30:d=8", "-vf", chain, "-c:v", "libx264",
            "-preset", "ultrafast", *(["-fps_mode", "vfr"] if vfr else []), str(path))
    return path


def _first_flash(path: Path) -> float:
    with FrameExtractor(path) as extractor:
        for timestamp, frame in extractor.frames_in_range(0, 8, sample_fps=30):
            if frame.mean() > 200:
                return timestamp
    raise AssertionError("вспышка не найдена")


def test_vfr_is_detected_and_cfr_video_is_not(tmp_path):
    vfr = probe_frame_timing(_flash_video(tmp_path / "vfr.mp4", vfr=True))
    cfr = probe_frame_timing(_flash_video(tmp_path / "cfr.mp4", vfr=False))
    assert vfr.needs_cfr_proxy and vfr.max_drift_sec > 0.5
    assert not cfr.needs_cfr_proxy and cfr.max_drift_sec < 0.01
    assert vfr.proxy_fps == 30


def test_cfr_proxy_puts_frames_back_at_their_real_time(tmp_path):
    vfr = _flash_video(tmp_path / "vfr.mp4", vfr=True)
    assert abs(_first_flash(vfr) - FLASH_AT) > 0.5                    # так анализ видел VFR раньше: время съехало
    proxy = make_cfr_proxy(vfr, tmp_path / "proxy.mp4", probe_frame_timing(vfr).proxy_fps)
    assert _first_flash(proxy) == pytest.approx(FLASH_AT, abs=0.04)    # копия: кадр на своём месте


def test_rotated_phone_video_has_upright_size_and_frames(tmp_path):
    lying = tmp_path / "lying.mp4"
    # вертикальный кадр 180x320 с белой полосой сверху, сохранённый лёжа (320x180) с тегом поворота
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=180x320:r=30:d=2", "-vf",
            "drawbox=x=0:y=0:w=iw:h=40:color=white:t=fill,transpose=2", "-c:v", "libx264", "-preset", "ultrafast", str(lying))
    rotated = tmp_path / "rotated.mp4"
    _ffmpeg("-display_rotation", "-90", "-i", str(lying), "-c", "copy", str(rotated))

    source = FFmpegWrapper().get_video_source(rotated)
    assert (source.width, source.height) == (180, 320)
    with FrameExtractor(rotated) as extractor:
        frame = extractor.frame_at(1.0)
    assert frame.shape[:2] == (320, 180)
    assert np.mean(frame[:30]) > 200 and np.mean(frame[-100:]) < 50    # полоса сверху: кадр стоит, а не лежит
