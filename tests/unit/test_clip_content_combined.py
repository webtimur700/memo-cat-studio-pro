"""Единый JSON-запрос на тексты клипа: разбор ответа, откаты на отдельные запросы, приоритет замка."""

from __future__ import annotations

import json
import threading
import time

import pytest

from core.priority_lock import PriorityLock
from llm.content_generator import generate_clip_content
from llm.lm_studio_provider import LLMRequestError, LLMUnavailableError
from llm.prompts.clip_content_prompt import parse_clip_content

GOOD = {
    "titles": [f"Заголовок {i}" for i in range(1, 11)],
    "description": "Кот роняет чашку.",
    "hashtags": ["#cats", "dogs", "#Cats", "#котики"],
}


class FakeProvider:
    """Отвечает на complete_json заданным ответом, на обычные запросы — каноническими заглушками; ведёт журнал."""

    def __init__(self, json_reply=None, json_error: Exception | None = None) -> None:
        self.json_reply, self.json_error = json_reply, json_error
        self.calls: list[str] = []
        self.images: list[bool] = []

    def complete_json(self, system_prompt, user_prompt, schema, max_tokens=2048, images=None, schema_name="response"):
        self.calls.append("json")
        self.images.append(bool(images))
        if self.json_error is not None:
            raise self.json_error
        return self.json_reply if isinstance(self.json_reply, str) else json.dumps(self.json_reply)

    def complete(self, system_prompt, user_prompt, max_tokens=512, images=None):
        self.images.append(bool(images))
        if "нумерованным списком" in system_prompt:
            self.calls.append("titles")
            return "\n".join(f"{i}. Отдельный {i}" for i in range(1, 11))
        if "СТРОГО списком хештегов" in system_prompt:
            self.calls.append("hashtags")
            return "#a #b #c"
        self.calls.append("description")
        return "Отдельное описание."

    def list_models(self):
        return ["fake"]


def test_combined_is_one_request_and_result_is_cleaned():
    provider = FakeProvider(GOOD)
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["json"]
    assert content.requests == 1 and not content.errors
    assert content.titles == tuple(GOOD["titles"])
    assert content.description == "Кот роняет чашку."
    assert content.hashtags == ("#cats", "#dogs", "#котики")     # # добавлен, дубль без учёта регистра убран


def test_separate_mode_still_makes_three_requests():
    provider = FakeProvider(GOOD)
    content = generate_clip_content(provider, "привет", combined=False)
    assert provider.calls == ["titles", "description", "hashtags"]
    assert content.requests == 3


def test_provider_without_json_support_falls_back_to_three_requests():
    class Plain(FakeProvider):
        complete_json = None

    provider = Plain()
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["titles", "description", "hashtags"]
    assert len(content.titles) == 10 and not content.errors


def test_garbage_reply_falls_back_without_reporting_an_error():
    provider = FakeProvider("это не JSON")
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["json", "titles", "description", "hashtags"]
    assert not content.errors and content.description == "Отдельное описание."


def test_only_missing_parts_are_requested_again():
    provider = FakeProvider({**GOOD, "hashtags": []})
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["json", "hashtags"]
    assert content.hashtags == ("#a", "#b", "#c") and content.titles == tuple(GOOD["titles"])


def test_server_rejects_schema_then_three_requests_work():
    provider = FakeProvider(json_error=LLMRequestError("HTTP 400 response_format"))
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["json", "titles", "description", "hashtags"]
    assert not content.errors


def test_unavailable_server_stops_after_first_request():
    provider = FakeProvider(json_error=LLMUnavailableError("нет соединения"))
    content = generate_clip_content(provider, "привет")
    assert provider.calls == ["json"]
    assert content.is_empty and len(content.errors) == 1


def test_image_is_dropped_when_model_rejects_it_and_request_repeated():
    class NoVision(FakeProvider):
        def complete_json(self, *args, images=None, **kwargs):
            if images:
                raise LLMRequestError("image not supported")
            return super().complete_json(*args, images=images, **kwargs)

    provider = NoVision(GOOD)
    content = generate_clip_content(provider, "привет", image_jpeg=b"\xff\xd8")
    assert provider.calls == ["json"] and content.requests == 2 and not content.used_image
    assert len(content.titles) == 10


def test_image_reaches_the_single_request():
    provider = FakeProvider(GOOD)
    content = generate_clip_content(provider, "привет", image_jpeg=b"\xff\xd8")
    assert provider.images == [True] and content.used_image


def test_parse_tolerates_markdown_fence_and_numbering():
    raw = "```json\n" + json.dumps({**GOOD, "titles": [f"{i}. \"Т{i}\"" for i in range(1, 12)]}) + "\n```"
    parsed = parse_clip_content(raw)
    assert parsed.titles[0] == "Т1" and len(parsed.titles) == 10


def test_hashtags_as_one_string_are_split_and_deduplicated():
    parsed = parse_clip_content(json.dumps({**GOOD, "hashtags": "#cats #dogs, #Cats  котики #shorts"}))
    assert parsed.hashtags == ("#cats", "#dogs", "#котики", "#shorts")


def test_schema_bounds_hashtags_length():
    from llm.prompts.clip_content_prompt import CLIP_CONTENT_SCHEMA

    tags = CLIP_CONTENT_SCHEMA["properties"]["hashtags"]
    assert tags["type"] == "string" and tags["maxLength"] > tags["minLength"] > 0   # без верхней границы генерация может не остановиться


def test_parse_rejects_non_object():
    with pytest.raises(ValueError):
        parse_clip_content("[1, 2]")


def test_urgent_overtakes_queued_background_work():
    lock = PriorityLock()
    order: list[str] = []
    first_running = threading.Event()

    def worker(name: str, urgent: bool, hold: float) -> None:
        ctx = lock.urgent() if urgent else lock.background()
        with ctx:
            order.append(name)
            if name == "b0":
                first_running.set()
            time.sleep(hold)

    threads = [threading.Thread(target=worker, args=("b0", False, 0.3))]
    threads[0].start()
    first_running.wait()
    for name, urgent in (("b1", False), ("b2", False), ("translate", True)):
        t = threading.Thread(target=worker, args=(name, urgent, 0.05))
        threads.append(t)
        t.start()
        time.sleep(0.05)   # порядок постановки в очередь: b1, b2, translate
    for t in threads:
        t.join()
    assert order[:2] == ["b0", "translate"] and set(order) == {"b0", "b1", "b2", "translate"}


def test_prompt_fixes_language_per_field():
    """Регресс: общая фраза «на английском» превращала в английские и заголовки, и описание."""
    from llm.prompts.clip_content_prompt import SYSTEM_PROMPT

    assert "Заголовки и описание пиши на РУССКОМ" in SYSTEM_PROMPT and "хештеги — на английском" in SYSTEM_PROMPT
