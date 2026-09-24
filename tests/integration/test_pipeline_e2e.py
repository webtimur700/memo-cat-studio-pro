"""Сквозной прогон PipelineRunner на коротком видео: клип + обложка + JSON."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from core.entities.settings import ShortsSettings, UserSettings, ViralScoreSettings
from core.entities.subtitle import WordTiming
from pipeline.pipeline_runner import PipelineRunner


class _StubLLM:
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        if "нумерованным списком" in system_prompt:
            return "\n".join(f"{i}. Заголовок {i} для теста" for i in range(1, 11))
        if "Отвечай СТРОГО списком хештегов" in system_prompt:
            return " ".join(f"#тег{i}" for i in range(1, 40))
        return "SEO-описание про кота."

    def list_models(self) -> list[str]:
        return ["stub"]


class _StubTranscriber:
    def __init__(self, **_kwargs):
        pass

    def transcribe(self, audio_path: Path):
        return [WordTiming("привет", 0.2, 0.7), WordTiming("мир", 0.8, 1.3)]


@pytest.fixture()
def short_video(tmp_path) -> Path:
    path = tmp_path / "short.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:duration=6:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def _settings() -> UserSettings:
    base = UserSettings()
    return replace(
        base,
        shorts=ShortsSettings(allowed_durations_sec=(5,), min_duration_sec=5, max_duration_sec=5),
        viral_score=ViralScoreSettings(queue_threshold=0),
    )


def test_pipeline_produces_clip_cover_and_metadata(short_video, tmp_path, monkeypatch):
    import subtitles.subtitle_service as subtitle_service

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _StubTranscriber)
    out = tmp_path / "out"

    clips = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=out, llm_provider=_StubLLM()).process_video(
        short_video, _settings()
    )

    assert len(clips) >= 1
    clip = clips[0]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(clip.output_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "1080,1920"

    assert clip.cover_path is not None and clip.cover_path.name == f"{clip.output_path.stem}_cover.png"
    with Image.open(clip.cover_path) as cover:
        assert cover.size == (1080, 1920)

    data = json.loads(clip.metadata_path.read_text(encoding="utf-8"))
    assert clip.metadata_path.name == f"{clip.output_path.stem}.json"
    assert len(data["titles"]) == 10 and data["title"] == "Заголовок 1 для теста"
    assert len(data["hashtags"]) == 30
    assert data["transcript"] == "привет мир"
    assert data["cover_file"] == clip.cover_path.name
    assert list(out.glob("_tmp_*")) == []

    # логотип (фиолетовый) реально запечён в клип: кадр на 2 с, пиксель в центре тела логотипа
    from effects.branding_overlay import logo_xy

    frame_path = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", str(clip.output_path), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    with Image.open(frame_path) as frame:
        logo_w, logo_h = int(1080 * 0.19), int(1080 * 0.19 * 240 / 640)
        x, y = logo_xy("top_right", (1080, 1920), (logo_w, logo_h))
        r, g, b = frame.convert("RGB").getpixel((x + 8, y + logo_h - 10))
    assert (abs(r - 124) < 30, abs(g - 92) < 30, abs(b - 255) < 30) == (True, True, True)


def test_pipeline_degrades_without_llm_and_models(short_video, tmp_path, monkeypatch):
    import subtitles.subtitle_service as subtitle_service

    class _Broken:
        def __init__(self, **_kwargs):
            raise RuntimeError("model missing")

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _Broken)
    clips = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=tmp_path / "out").process_video(
        short_video, _settings()
    )
    assert clips and clips[0].output_path.exists()
    assert clips[0].title.startswith("Момент ")
    assert clips[0].titles == ()
