"""HDR-исходник (HLG как у iPhone, PQ как HDR10): клип, обложка и кадр для LLM должны быть SDR с цветами исходника,
а не блёклыми 10-битными значениями, прочитанными как BT.709."""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.entities.settings import ExportSettings
from export.dynamic_crop import CropSample
from export.export_service import ExportPlan, ExportService
from video.hdr import hdr_transfer, tonemapped_frame
from vision.smart_crop import CropWindow

W, H = 320, 180


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def sdr(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("hdr") / "sdr.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r=30:d=3", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(path))
    return path


def _to_hdr(sdr: Path, transfer: str) -> Path:
    """Тот же ролик в HDR: белый SDR = 203 нит (BT.2408), BT.2020, 10 бит HEVC — как в scripts/make_test_videos.py."""
    out = sdr.with_name(f"{transfer}.mp4")
    if not out.exists():
        _ffmpeg("-i", str(sdr), "-vf",
                f"zscale=tin=bt709:min=bt709:pin=bt709:rin=tv:t=linear:npl=203,format=gbrpf32le,"
                f"zscale=p=bt2020:t={transfer}:m=bt2020nc:r=tv:npl=203,format=yuv420p10le",
                "-c:v", "libx265", "-preset", "ultrafast", "-crf", "12", "-x265-params",
                f"log-level=error:colorprim=bt2020:transfer={transfer}:colormatrix=bt2020nc",
                "-color_primaries", "bt2020", "-color_trc", transfer, "-colorspace", "bt2020nc", str(out))
    return out


def _frame(path: Path, t: float = 1.0) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)


def _saturation(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())


@pytest.mark.parametrize("transfer", ["arib-std-b67", "smpte2084"])
def test_hdr_is_detected_and_tonemapped_frame_matches_the_sdr_original(sdr, transfer):
    hdr = _to_hdr(sdr, transfer)
    assert hdr_transfer(sdr) is None and hdr_transfer(hdr) == transfer
    original = _frame(sdr).astype(np.float32)
    washed = _frame(hdr)                                       # как читают ffmpeg/OpenCV без тонмаппинга
    mapped = tonemapped_frame(hdr, 1.0, W, H).astype(np.float32)
    assert np.abs(mapped - original).mean() < 12      # погрешность самого перевода 709 -> 2020/10 бит -> 709 на testsrc2 ≈ 8.5
    assert np.abs(washed.astype(np.float32) - original).mean() > 2 * np.abs(mapped - original).mean()


@pytest.mark.parametrize("transfer", ["arib-std-b67", "smpte2084"])
def test_export_of_hdr_source_is_sdr_bt709_with_original_colors(sdr, transfer, tmp_path):
    hdr = _to_hdr(sdr, transfer)
    crop = [CropSample(float(i), CropWindow(x1=110, y1=0, x2=211, y2=180, zoom_factor=1.0, timestamp_sec=float(i))) for i in range(3)]

    def export(source: Path, name: str) -> Path:
        plan = ExportPlan(source_path=source, output_path=tmp_path / name, crop_samples=crop, subtitle_ass_path=None,
                          banner_rect=None, banner_text_lines=[], banner_appear_at_sec=0.0, banner_duration_sec=0.0,
                          settings=ExportSettings(width=270, height=480, fps=30, quality_preset="high", encoder="x264"))
        return ExportService().export_clip(plan)

    from_sdr, from_hdr = export(sdr, "sdr_clip.mp4"), export(hdr, "hdr_clip.mp4")
    trc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_transfer,pix_fmt",
                          "-of", "default=nw=1", str(from_hdr)], capture_output=True, text=True, check=True).stdout
    assert "pix_fmt=yuv420p\n" in trc and "smpte2084" not in trc and "arib-std-b67" not in trc

    def frame(clip: Path) -> np.ndarray:
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "1", "-i", str(clip), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).reshape(480, 270, 3)

    a, b = frame(from_sdr), frame(from_hdr)
    assert np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() < 12
    assert _saturation(b) > 0.9 * _saturation(a)
