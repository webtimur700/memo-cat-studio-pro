import sys
import types
import wave
from pathlib import Path

from core.entities.moment import Moment
from core.entities.settings import UserSettings
from core.entities.subtitle import WordTiming
from pipeline.pipeline_runner import PipelineRunner
from subtitles.subtitle_service import WhisperTranscriber
from video.ffmpeg_wrapper import FFmpegWrapper


def _fake_faster_whisper(monkeypatch, fail_offline: bool):
    calls: list[dict] = []

    class FakeModel:
        def __init__(self, size, device="cpu", compute_type="int8", local_files_only=False):
            calls.append({"local_files_only": local_files_only})
            if local_files_only and fail_offline:
                raise RuntimeError("not in cache")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


def test_model_loaded_from_local_cache_first(monkeypatch):
    calls = _fake_faster_whisper(monkeypatch, fail_offline=False)
    WhisperTranscriber()._ensure_model_loaded()
    assert calls == [{"local_files_only": True}]


def test_model_falls_back_to_network_when_not_cached(monkeypatch):
    calls = _fake_faster_whisper(monkeypatch, fail_offline=True)
    WhisperTranscriber()._ensure_model_loaded()
    assert calls == [{"local_files_only": True}, {"local_files_only": False}]


def test_model_loaded_once_per_transcriber(monkeypatch):
    calls = _fake_faster_whisper(monkeypatch, fail_offline=False)
    transcriber = WhisperTranscriber()
    transcriber._ensure_model_loaded()
    transcriber._ensure_model_loaded()
    assert len(calls) == 1


def test_extract_audio_track_only_requested_segment(sample_video, tmp_path):
    out = FFmpegWrapper().extract_audio_track(
        sample_video, tmp_path / "seg.wav", start_sec=5.0, duration_sec=3.0
    )
    with wave.open(str(out)) as wav:
        duration = wav.getnframes() / wav.getframerate()
    assert abs(duration - 3.0) < 0.1


def test_pipeline_shares_one_transcriber_across_moments(sample_video, tmp_path, monkeypatch):
    created: list[object] = []

    class FakeTranscriber:
        def __init__(self, **_kwargs):
            created.append(self)
            self.durations: list[float] = []

        def transcribe(self, audio_path: Path):
            with wave.open(str(audio_path)) as wav:
                self.durations.append(wav.getnframes() / wav.getframerate())
            return [WordTiming("привет", 0.1, 0.5)]

    import subtitles.subtitle_service as subtitle_service

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", FakeTranscriber)

    runner = PipelineRunner(output_dir=tmp_path)
    settings = UserSettings()
    moment_a = Moment(0.0, 5.0, 80, 0.5, ())
    moment_b = Moment(10.0, 15.0, 80, 0.5, ())

    words_a = runner._transcribe_moment(sample_video, moment_a, settings)
    words_b = runner._transcribe_moment(sample_video, moment_b, settings)

    assert len(created) == 1
    # транскрибируется звук момента (5 с), а не всего 20-секундного видео
    assert all(abs(d - 5.0) < 0.2 for d in created[0].durations)
    assert words_a and words_b
    assert list(tmp_path.glob("_tmp_audio_*")) == []


def test_transcription_failure_degrades_to_no_words(sample_video, tmp_path, monkeypatch):
    runner = PipelineRunner(output_dir=tmp_path)
    monkeypatch.setattr(runner, "_get_transcriber", lambda _s: (_ for _ in ()).throw(RuntimeError("no model")))
    assert runner._transcribe_moment(sample_video, Moment(0.0, 5.0, 80, 0.5, ()), UserSettings()) == []
