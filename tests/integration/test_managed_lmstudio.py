import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm.lm_studio_provider import LLMUnavailableError, LMStudioConfig
from llm.managed_provider import ManagedLMStudio
from llm.model_selector import ModelProfile, select_model  # noqa: F401
import llm.model_selector as model_selector


class _State:
    def __init__(self):
        self.loaded: list[str] = []
        self.load_calls: list[dict] = []
        self.unload_calls: list[str] = []
        self.chat_bodies: list[dict] = []
        self.admin_api = True


def _models_payload(state: _State):
    def entry(key, kind, size, vision, reasoning):
        e = {"type": kind, "key": key, "size_bytes": size, "loaded_instances": [{"identifier": key}] if key in state.loaded else [],
             "max_context_length": 262144, "capabilities": {"vision": vision}}
        if reasoning:
            e["capabilities"]["reasoning"] = {"allowed_options": ["off", "on"], "default": "on"}
        return e
    return {"models": [
        entry("nomic-embed", "embedding", 84_000_000, False, False),
        entry("vendor/small-chat", "llm", 4_000_000_000, False, False),
        entry("vendor/big-vision", "llm", 12_000_000_000, True, True),
    ]}


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/v1/models" and state.admin_api:
                self._send(_models_payload(state))
            elif self.path == "/v1/models":
                self._send({"data": [{"id": m["key"]} for m in _models_payload(state)["models"]]})
            else:
                self._send({"error": "no"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/v1/models/load" and state.admin_api:
                state.load_calls.append(body)
                state.loaded.append(body["model"])
                self._send({"status": "loaded", "instance_id": body["model"]})
            elif self.path == "/api/v1/models/unload" and state.admin_api:
                state.unload_calls.append(body["instance_id"])
                state.loaded = [m for m in state.loaded if m != body["instance_id"]]
                self._send({"instance_id": body["instance_id"]})
            elif self.path == "/v1/chat/completions":
                state.chat_bodies.append(body)
                self._send({"choices": [{"message": {"content": "1. Заголовок"}}], "usage": {"completion_tokens": 3}})
            else:
                self._send({"error": "no"}, 404)
    return Handler


@pytest.fixture()
def lm_server():
    state = _State()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    monkeypatch.setattr(model_selector, "DEFAULT_PROFILES", (
        ModelProfile("big-vision", 1, "none"), ModelProfile("small-chat", 2, None),
    ))


def _managed(url, available_mib, **config):
    return ManagedLMStudio(LMStudioConfig(base_url=url, **config), reserve_mib=1024, mem_reader=lambda: available_mib)


def test_picks_best_fitting_model_loads_it_with_small_context_and_uses_effort(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024)
    assert managed.complete("s", "u") == "1. Заголовок"
    assert managed.selection.model_key == "vendor/big-vision"
    assert state.load_calls == [{"model": "vendor/big-vision", "context_length": 8192, "parallel": 1, "flash_attention": True}]
    body = state.chat_bodies[0]
    assert body["model"] == "vendor/big-vision" and body["reasoning_effort"] == "none"
    assert managed.supports_vision is True


def test_falls_back_to_smaller_model_when_memory_is_short(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=8 * 1024)      # 12 ГБ vision-модель не влезает
    managed.complete("s", "u")
    assert managed.selection.model_key == "vendor/small-chat"
    assert "reasoning_effort" not in state.chat_bodies[0]
    assert managed.supports_vision is False


def test_override_wins_and_embedding_is_never_used(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024, model_override="vendor/small-chat")
    managed.complete("s", "u")
    assert state.load_calls[0]["model"] == "vendor/small-chat"
    assert "embed" not in state.chat_bodies[0]["model"]


def test_foreign_loaded_model_is_unloaded_and_already_loaded_choice_is_not_reloaded(lm_server):
    state, url = lm_server
    state.loaded = ["vendor/small-chat"]
    managed = _managed(url, available_mib=64 * 1024)
    managed.complete("s", "u")
    assert state.unload_calls == ["vendor/small-chat"] and state.loaded == ["vendor/big-vision"]

    state.load_calls.clear()
    again = _managed(url, available_mib=64 * 1024)
    again.complete("s", "u")
    assert state.load_calls == []                        # уже загружена — второй раз не грузим


def test_release_unloads_only_what_we_loaded(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024)
    managed.complete("s", "u")
    managed.release()
    assert state.loaded == []


def test_image_is_sent_as_data_url(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024)
    managed.complete("s", "u", images=[b"\xff\xd8jpeg"])
    content = state.chat_bodies[0]["messages"][1]["content"]
    assert content[0]["type"] == "text" and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_background_loading_then_complete_waits_for_it(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024)
    managed.start_loading_in_background()
    managed.complete("s", "u")
    assert len(state.load_calls) == 1


def test_older_lm_studio_without_rest_admin_falls_back_to_jit(lm_server):
    state, url = lm_server
    state.admin_api = False
    managed = _managed(url, available_mib=64 * 1024)
    assert managed.complete("s", "u") == "1. Заголовок"      # /v1/models + JIT: приложение не падает
    assert state.load_calls == []


def test_unreachable_server_raises_unavailable():
    managed = _managed("http://127.0.0.1:1/v1", available_mib=64 * 1024)
    with pytest.raises(LLMUnavailableError):
        managed.complete("s", "u")
    assert managed.supports_vision is False


def test_manage_models_off_behaves_like_plain_provider(lm_server):
    state, url = lm_server
    managed = _managed(url, available_mib=64 * 1024, manage_models=False, model_override="vendor/small-chat")
    managed.complete("s", "u")
    assert state.load_calls == []
