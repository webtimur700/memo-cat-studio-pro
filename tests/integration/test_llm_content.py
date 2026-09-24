import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from core.entities.moment import Moment
from llm.content_generator import build_clip_description, generate_clip_content
from llm.lm_studio_provider import (
    LLMRequestError,
    LLMUnavailableError,
    LMStudioConfig,
    LMStudioProvider,
    strip_reasoning,
)
from pipeline.pipeline_runner import PipelineRunner

TITLES = "\n".join(f"{i}. Заголовок номер {i}" for i in range(1, 11))
HASHTAGS = " ".join(f"#tag{i}" for i in range(1, 36))


class _ReasoningHandler(BaseHTTPRequestHandler):
    """Имитирует reasoning-модель: перед ответом всегда идёт <think>…</think>."""

    prompts: list[str] = []

    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "qwen3.6-test"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = request["messages"][0]["content"], request["messages"][1]["content"]
        type(self).prompts.append(user)
        if "нумерованным списком" in system:
            answer = TITLES
        elif "хештег" in system.lower() and "Отвечай СТРОГО списком" in system:
            answer = HASHTAGS
        else:
            answer = "Кот и его хозяйка. Смешное видео про котов."
        self._send({"choices": [{"message": {"content": f"<think>Сначала подумаю: #мысль 1. не ответ</think>\n\n{answer}"}}]})


@pytest.fixture()
def reasoning_server():
    _ReasoningHandler.prompts = []
    server = HTTPServer(("127.0.0.1", 0), _ReasoningHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_strip_reasoning_removes_closed_block():
    assert strip_reasoning("<think>\nрассуждаю\n</think>\n\nОтвет") == "Ответ"


def test_strip_reasoning_unclosed_block_means_truncated():
    assert strip_reasoning("<think>рассуждаю и рассуждаю…") == ""


def test_strip_reasoning_only_closing_tag():
    assert strip_reasoning("мысли</think>Ответ") == "Ответ"


def test_strip_reasoning_plain_text_untouched():
    assert strip_reasoning("  просто ответ ") == "просто ответ"


def test_provider_strips_think_and_content_generated(reasoning_server):
    provider = LMStudioProvider(LMStudioConfig(base_url=reasoning_server))
    content = generate_clip_content(provider, "ты что встал ты что встал")
    assert len(content.titles) == 10
    assert content.titles[0] == "Заголовок номер 1"
    assert content.description.startswith("Кот и его хозяйка")
    # хештеги из блока <think> (#мысль) не просочились, и не больше 30
    assert len(content.hashtags) == 30
    assert "#мысль" not in content.hashtags
    assert not content.errors
    # в LLM ушёл текст транскрипции, а не "Момент с viral score"
    assert all("ты что встал" in p for p in _ReasoningHandler.prompts)


def test_truncated_reasoning_raises_clear_error(monkeypatch):
    provider = LMStudioProvider(LMStudioConfig(model_override="m"))

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "<think>бесконечно думаю"}}]}).encode()

    import llm.lm_studio_provider as mod

    monkeypatch.setattr(mod._LOCAL_OPENER, "open", lambda *a, **k: _Resp())
    with pytest.raises(LLMRequestError, match="max_tokens"):
        provider.complete("s", "u")


def test_no_speech_description_is_honest():
    assert "нет речи" in build_clip_description("   ")


def test_unavailable_server_stops_after_first_failure():
    provider = LMStudioProvider(LMStudioConfig(base_url="http://127.0.0.1:1/v1", timeout_sec=2.0))
    content = generate_clip_content(provider, "текст")
    assert content.is_empty
    assert len(content.errors) == 1  # не три таймаута подряд


def test_from_env_reads_dotenv(tmp_path, monkeypatch):
    for key in ("LM_STUDIO_BASE_URL", "LM_STUDIO_REQUEST_TIMEOUT_SEC", "LM_STUDIO_MODEL_OVERRIDE"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("LM_STUDIO_BASE_URL=http://host:9999/v1\nLM_STUDIO_REQUEST_TIMEOUT_SEC=7\nLM_STUDIO_MODEL_OVERRIDE=\n")
    config = LMStudioConfig.from_env(env)
    assert config.base_url == "http://host:9999/v1"
    assert config.timeout_sec == 7.0
    assert config.model_override is None
    for key in ("LM_STUDIO_BASE_URL", "LM_STUDIO_REQUEST_TIMEOUT_SEC"):
        monkeypatch.delenv(key, raising=False)


def test_runner_writes_metadata_json_and_default_title_without_llm(tmp_path):
    runner = PipelineRunner(output_dir=tmp_path, llm_provider=None)
    moment = Moment(5.0, 20.0, 75, 0.5, ())
    assert runner._generate_content("привет").is_empty
    path = runner._write_metadata(
        tmp_path / "clip1.mp4", Path("src.mp4"), moment, "привет", runner._default_title(moment),
        runner._generate_content("привет"),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "clip1.json"
    assert data["title"] == "Момент 5s (score 75)"
    assert data["titles"] == [] and data["transcript"] == "привет"


def test_runner_with_llm_fills_content_and_marks_unavailable_once(reasoning_server):
    runner = PipelineRunner(llm_provider=LMStudioProvider(LMStudioConfig(base_url=reasoning_server)))
    content = runner._generate_content("привет мир")
    assert len(content.titles) == 10

    dead = PipelineRunner(llm_provider=LMStudioProvider(LMStudioConfig(base_url="http://127.0.0.1:1/v1", timeout_sec=2)))
    assert dead._generate_content("x").is_empty
    assert dead._llm_unavailable is True
