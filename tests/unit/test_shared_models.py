from pathlib import Path

from pipeline.pipeline_runner import PipelineRunner
from pipeline.shared_models import SharedModels


class _FakeTranscriber:
    created = 0

    def __init__(self, **_kwargs):
        type(self).created += 1


def test_transcriber_created_once_for_many_runners(monkeypatch, tmp_path):
    import subtitles.subtitle_service as subtitle_service

    _FakeTranscriber.created = 0
    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _FakeTranscriber)
    shared = SharedModels(tmp_path)
    runners = [PipelineRunner(models_dir=tmp_path, output_dir=tmp_path, shared_models=shared) for _ in range(3)]
    from core.entities.settings import UserSettings

    settings = UserSettings()
    got = [r._get_transcriber(settings) for r in runners for _ in range(2)]
    assert _FakeTranscriber.created == 1 and shared.transcriber_loads == 1
    assert all(t is got[0] for t in got)


def test_close_releases_models_so_next_batch_reloads(monkeypatch, tmp_path):
    import subtitles.subtitle_service as subtitle_service

    _FakeTranscriber.created = 0
    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _FakeTranscriber)
    shared = SharedModels(tmp_path)
    shared.transcriber("small", "int8")
    shared.close()
    shared.transcriber("small", "int8")
    assert _FakeTranscriber.created == 2


def test_missing_yolo_returns_none_and_is_not_retried(tmp_path):
    shared = SharedModels(tmp_path / "nope")
    assert shared.detector() is None and shared.detector() is None
