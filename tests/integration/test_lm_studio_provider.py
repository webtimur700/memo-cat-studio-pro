import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm.lm_studio_provider import LLMRequestError, LMStudioConfig, LMStudioProvider


class _MockLMStudioHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json({"data": [{"id": "qwen/qwen3.6-35b-a3b"}, {"id": "google/gemma-4-e4b"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers["Content-Length"])
            request_body = json.loads(self.rfile.read(length))
            reply = f"Модель {request_body['model']} отвечает на: {request_body['messages'][1]['content'][:30]}"
            self._send_json({"choices": [{"message": {"content": reply}}]})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def mock_lm_studio_url():
    server = HTTPServer(("127.0.0.1", 0), _MockLMStudioHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


def test_list_models(mock_lm_studio_url):
    provider = LMStudioProvider(LMStudioConfig(base_url=mock_lm_studio_url))
    models = provider.list_models()
    assert models == ["qwen/qwen3.6-35b-a3b", "google/gemma-4-e4b"]


def test_complete_auto_selects_first_model(mock_lm_studio_url):
    provider = LMStudioProvider(LMStudioConfig(base_url=mock_lm_studio_url))
    response = provider.complete("system", "Придумай заголовок")
    assert "qwen/qwen3.6-35b-a3b" in response


def test_complete_respects_model_override(mock_lm_studio_url):
    provider = LMStudioProvider(LMStudioConfig(base_url=mock_lm_studio_url, model_override="google/gemma-4-e4b"))
    response = provider.complete("system", "тест")
    assert "google/gemma-4-e4b" in response


def test_unreachable_server_raises_clear_error():
    provider = LMStudioProvider(LMStudioConfig(base_url="http://127.0.0.1:1", timeout_sec=2.0))
    with pytest.raises(LLMRequestError):
        provider.list_models()
