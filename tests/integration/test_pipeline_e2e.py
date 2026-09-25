"""Сквозной прогон PipelineRunner на коротком видео: клип + обложка + JSON."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from core.entities.settings import AudioSettings, ShortsSettings, UserSettings, ViralScoreSettings
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

    def transcribe(self, audio_path: Path, language=None):
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
        audio=AudioSettings(music_enabled=False),   # музыка проверяется отдельными тестами, не берём настоящую assets/music
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
    from effects.safe_zone import SafeZone

    frame_path = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", str(clip.output_path), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    with Image.open(frame_path) as frame:
        logo_w, logo_h = int(1080 * 0.19), int(1080 * 0.19 * 240 / 640)
        x, y = logo_xy("top_right", SafeZone.from_settings((1080, 1920), _settings().safe_zone), (logo_w, logo_h))
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


def _run_with_stub_words(short_video, tmp_path, monkeypatch, settings):
    import subtitles.subtitle_service as subtitle_service

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _StubTranscriber)
    out = tmp_path / "out"
    clips = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=out, llm_provider=_StubLLM()).process_video(
        short_video, settings
    )
    return out, clips


def test_srt_and_ass_are_saved_next_to_the_clip_with_timing_from_clip_start(short_video, tmp_path, monkeypatch):
    out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, _settings())
    clip = clips[0]
    srt, ass = clip.output_path.with_suffix(".srt"), clip.output_path.with_suffix(".ass")
    assert set(clip.subtitle_paths) == {srt, ass}

    srt_text = srt.read_text(encoding="utf-8")
    assert srt_text.startswith("1\n00:00:00,200 --> 00:00:01,300\nпривет мир\n")   # от начала клипа, не исходника
    ass_text = ass.read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.20,0:00:01.30" in ass_text and "PlayResY: 1920" in ass_text

    meta = json.loads(clip.metadata_path.read_text(encoding="utf-8"))
    assert meta["subtitle_files"] == {"srt": srt.name, "ass": ass.name}
    assert list(out.glob("_tmp_*")) == []


def test_subtitle_files_are_saved_even_without_burn_in(short_video, tmp_path, monkeypatch):
    settings = _settings().with_field("subtitles", burn_in=False)
    out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, settings)
    assert clips[0].output_path.with_suffix(".srt").exists() and clips[0].output_path.with_suffix(".ass").exists()


def test_only_requested_formats_are_kept_and_burn_in_leaves_no_temp_ass(short_video, tmp_path, monkeypatch):
    settings = _settings().with_field("subtitles", export_formats=("srt",))
    out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, settings)
    assert clips[0].output_path.with_suffix(".srt").exists()
    assert not clips[0].output_path.with_suffix(".ass").exists()
    assert list(out.glob("*.ass")) == [] and list(out.glob("_tmp_*")) == []


def test_no_speech_means_no_subtitle_files(short_video, tmp_path, monkeypatch):
    import subtitles.subtitle_service as subtitle_service

    class _Silent(_StubTranscriber):
        def transcribe(self, audio_path, language=None):
            return []

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _Silent)
    out = tmp_path / "out"
    clips = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=out).process_video(short_video, _settings())
    assert clips and clips[0].subtitle_paths == () and list(out.glob("*.srt")) == []


def _music_track(tmp_path) -> Path:
    library = tmp_path / "music"
    library.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=1000:duration=10",
                    str(library / "song.wav")], check=True)
    return library


def _audio_stream_count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return len(out.split())


def test_music_from_library_is_mixed_in_and_recorded(short_video, tmp_path, monkeypatch):
    library = _music_track(tmp_path)
    settings = _settings().with_field("audio", music_enabled=True, music_library_path=str(library))
    out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, settings)
    assert _audio_stream_count(clips[0].output_path) == 1
    meta = json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))
    assert meta["music_file"] == "song.wav"
    assert list(out.glob("_tmp_*")) == []


def test_empty_music_folder_gives_clip_without_music_and_logs_reason(short_video, tmp_path, monkeypatch):
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        library = tmp_path / "empty_music"
        library.mkdir()
        settings = _settings().with_field("audio", music_enabled=True, music_library_path=str(library))
        out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, settings)
    finally:
        logger.remove(sink)
    assert clips and clips[0].output_path.exists()
    assert json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))["music_file"] is None
    assert any("Музыка не добавлена" in m for m in messages)


def test_music_disabled_in_settings_ignores_library(short_video, tmp_path, monkeypatch):
    library = _music_track(tmp_path)
    settings = _settings().with_field("audio", music_enabled=False, music_library_path=str(library))
    out, clips = _run_with_stub_words(short_video, tmp_path, monkeypatch, settings)
    assert json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))["music_file"] is None


class _DeadLLM:
    """Как ManagedLMStudio, которая не смогла подготовить модель: complete() бросает, причина — в .issue."""

    def __init__(self, issue):
        self.issue = issue
        self.calls = 0

    def complete(self, *args, **kwargs):
        from llm.lm_studio_provider import LLMUnavailableError

        self.calls += 1
        raise LLMUnavailableError("Не удалось подготовить модель LM Studio")

    def list_models(self):
        return []


def _issue():
    from core.entities.llm_issue import LLMIssue

    return LLMIssue("no_memory", "LLM не загружена: нужно ~14.4 ГиБ, свободно 13.3 ГиБ.", "Уменьшите запас в настройках.",
                    need_gib=14.4, free_gib=13.3, reserve_gib=6.0)


def test_llm_failure_is_reported_in_progress_clip_and_json_for_every_clip(short_video, tmp_path, monkeypatch):
    import subtitles.subtitle_service as subtitle_service

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _StubTranscriber)
    settings = _settings()
    llm = _DeadLLM(_issue())
    events = []
    runner = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=tmp_path / "out", llm_provider=llm)
    clips = runner.process_video(short_video, settings, progress=lambda stage, data: events.append((stage, data)))

    assert len(clips) >= 1
    warnings = [d for s, d in events if s == "llm_warning"]
    assert len(warnings) == 1 and warnings[0]["issue"]["need_gib"] == 14.4       # предупреждение один раз на видео
    calls = llm.calls
    assert calls >= 1
    later = runner._clip_llm_issue(runner._generate_content("ещё один клип того же видео", None))
    assert llm.calls == calls and later.kind == "no_memory"                     # после отказа LLM больше не дёргаем, причина остаётся
    for clip in clips:                                                          # причина есть у клипа
        assert clip.title.startswith("Момент ") and clip.llm_issue.kind == "no_memory"
        data = json.loads(clip.metadata_path.read_text(encoding="utf-8"))
        assert data["llm_issue"]["free_gib"] == 13.3 and "запас" in data["llm_issue"]["hint"]


def test_working_llm_leaves_no_issue(short_video, tmp_path, monkeypatch):
    import subtitles.subtitle_service as subtitle_service

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _StubTranscriber)
    events = []
    clips = PipelineRunner(models_dir=tmp_path / "no_models", output_dir=tmp_path / "out", llm_provider=_StubLLM()).process_video(
        short_video, _settings(), progress=lambda stage, data: events.append(stage)
    )
    assert clips[0].llm_issue is None and "llm_warning" not in events
    assert json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))["llm_issue"] is None
