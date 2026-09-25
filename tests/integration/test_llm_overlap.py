"""Тексты LLM клипа N считаются в фоне, пока кодируются следующие клипы (и идут строго по одному запросу к LLM)."""

import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from core.entities.settings import AudioSettings, ShortsSettings, UserSettings, ViralScoreSettings
from pipeline.pipeline_runner import PipelineRunner


class _SlowLLM:
    """Каждый запрос занимает `delay` секунд; считает одновременные запросы и порядок клипов."""

    def __init__(self, delay: float, fail_after: int | None = None):
        self.delay = delay
        self.fail_after = fail_after
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.first_call_at: float | None = None
        self._lock = threading.Lock()

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
            if self.first_call_at is None:
                self.first_call_at = time.perf_counter()
        try:
            time.sleep(self.delay)
            if "нумерованным списком" in system_prompt:
                return "\n".join(f"{i}. Заголовок {i} для теста" for i in range(1, 11))
            if "Отвечай СТРОГО списком хештегов" in system_prompt:
                return " ".join(f"#тег{i}" for i in range(1, 40))
            return "SEO-описание про кота."
        finally:
            with self._lock:
                self.active -= 1

    def list_models(self) -> list[str]:
        return ["stub"]


@pytest.fixture()
def long_video(tmp_path) -> Path:
    path = tmp_path / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:duration=40:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _settings() -> UserSettings:
    return replace(
        UserSettings(),
        shorts=ShortsSettings(allowed_durations_sec=(5,), min_duration_sec=5, max_duration_sec=5),
        viral_score=ViralScoreSettings(queue_threshold=0),
        audio=AudioSettings(music_enabled=False),
    )


@pytest.fixture(autouse=True)
def four_moments(monkeypatch):
    """Однородное тестовое видео даёт один момент: подставляем четыре фиксированных."""
    import pipeline.pipeline_runner as module
    from core.entities.moment import Moment

    monkeypatch.setattr(module, "select_moments", lambda *a, **k: [Moment(i * 9.0, i * 9.0 + 5.0, 50, 0.3) for i in range(4)])


def _run(video, out, llm):
    return PipelineRunner(models_dir=out / "no_models", output_dir=out, llm_provider=llm).process_video(video, _settings())


def test_llm_runs_in_background_one_request_at_a_time_and_results_keep_order(long_video, tmp_path):
    llm = _SlowLLM(delay=0.4)                 # 3 запроса на клип = 1.2 с на клип
    started = time.perf_counter()
    clips = _run(long_video, tmp_path / "out", llm)
    elapsed = time.perf_counter() - started

    assert len(clips) >= 3
    assert llm.max_active == 1                                              # не больше одного запроса к LLM за раз
    assert [c.moment.start_sec for c in clips] == sorted(c.moment.start_sec for c in clips)
    assert all(c.title == "Заголовок 1 для теста" and len(c.titles) == 10 for c in clips)
    assert all(c.metadata_path.exists() and c.cover_path is not None for c in clips)
    assert llm.calls == 3 * len(clips)


def test_llm_overlaps_with_encoding_of_the_next_clips(long_video, tmp_path):
    """Общее время меньше суммы «кодирование + LLM»: пока LLM занята клипом N, клип N+1 уже кодируется."""
    baseline_started = time.perf_counter()
    clips = _run(long_video, tmp_path / "base", _SlowLLM(delay=0.0))
    no_llm = time.perf_counter() - baseline_started
    n = len(clips)

    delay = 1.5                                # 4.5 с LLM на клип — заметно больше работы над клипом
    started = time.perf_counter()
    _run(long_video, tmp_path / "slow", _SlowLLM(delay=delay))
    with_llm = time.perf_counter() - started

    sequential = no_llm + 3 * delay * n       # так было бы, если бы LLM блокировала конвейер
    assert with_llm < sequential - 0.5 * (no_llm - no_llm / n) - 1.0, (with_llm, sequential)
    assert with_llm >= 3 * delay * n - 0.5    # быстрее, чем сами запросы, не бывает: они идут по очереди


def test_llm_failure_stops_further_requests_and_clips_still_finish(long_video, tmp_path):
    from llm.lm_studio_provider import LLMUnavailableError

    class _Dead(_SlowLLM):
        def complete(self, *args, **kwargs):
            super().complete(*args, **kwargs)
            raise LLMUnavailableError("LM Studio недоступна")

    llm = _Dead(delay=0.05)
    clips = _run(long_video, tmp_path / "out", llm)
    assert len(clips) >= 3 and all(c.titles == () and c.title.startswith("Момент ") for c in clips)
    assert llm.calls <= 2 and all(c.llm_issue is not None for c in clips)     # «не долбимся в недоступную LLM до конца видео»
