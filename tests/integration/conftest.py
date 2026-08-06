import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """Генерирует детерминированное тестовое видео 1280x720, 20с (синий 0-10с,
    красный 10-20с, тон 440Hz) — используется всеми интеграционными тестами,
    которым нужен реальный видеофайл на диске.
    """
    tmp_dir = tmp_path_factory.mktemp("video_fixtures")
    output_path = tmp_dir / "sample.mp4"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=10:r=30",
        "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=10:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
