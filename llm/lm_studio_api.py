"""Административный REST-клиент LM Studio: список моделей с метаданными,
загрузка (с ограниченным контекстом) и выгрузка.

Нужен потому, что OpenAI-совместимый /v1/models отдаёт только id: нельзя
отличить embedding-модель от чат-модели, узнать размер, vision и поддержку
режимов рассуждения. Нативный /api/v1/models (LM Studio 0.4+) всё это отдаёт.
Если его нет (старая версия) — деградируем до /v1/models с минимумом сведений.

Загрузка через REST важна: при JIT-загрузке (первый запрос к невыгруженной
модели) LM Studio берёт контекст по умолчанию — у этих моделей 262144 токена,
это гигабайты KV-кэша впустую. Мы грузим сами с context_length ~8-16К.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from core.exceptions import MemoCatError

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost — без прокси

GIB = 1024**3


class LMStudioAdminError(MemoCatError):
    pass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    key: str
    kind: str = "llm"                     # "llm" | "embedding"
    size_bytes: int | None = None
    params: str | None = None
    vision: bool = False
    reasoning_options: tuple[str, ...] = ()   # пусто — модель не рассуждающая
    loaded_instances: tuple[str, ...] = field(default_factory=tuple)
    max_context_length: int | None = None

    @property
    def is_chat_model(self) -> bool:
        return self.kind == "llm"

    @property
    def size_gib(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / GIB

    @property
    def supports_reasoning_control(self) -> bool:
        return bool(self.reasoning_options)


def api_root(base_url: str) -> str:
    """http://localhost:1234/v1 -> http://localhost:1234"""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _request(method: str, url: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LMStudioAdminError(f"{method} {url}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LMStudioAdminError(f"{method} {url}: {exc}") from exc


def parse_models_v1(payload: dict) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for item in payload.get("models", []):
        reasoning = (item.get("capabilities") or {}).get("reasoning") or {}
        models.append(
            ModelInfo(
                key=item["key"],
                kind=item.get("type", "llm"),
                size_bytes=item.get("size_bytes"),
                params=item.get("params_string"),
                vision=bool((item.get("capabilities") or {}).get("vision")),
                reasoning_options=tuple(reasoning.get("allowed_options", ())),
                loaded_instances=tuple(
                    inst.get("identifier") or inst.get("instance_id") or item["key"]
                    for inst in item.get("loaded_instances", [])
                ),
                max_context_length=item.get("max_context_length"),
            )
        )
    return models


def parse_models_openai(payload: dict) -> list[ModelInfo]:
    """Запасной вариант: только id; embedding отличаем по имени."""
    return [
        ModelInfo(key=item["id"], kind="embedding" if "embed" in item["id"].lower() else "llm")
        for item in payload.get("data", [])
    ]


def list_models(base_url: str, timeout: float = 10.0) -> list[ModelInfo]:
    root = api_root(base_url)
    try:
        return parse_models_v1(_request("GET", f"{root}/api/v1/models", None, timeout))
    except LMStudioAdminError:
        return parse_models_openai(_request("GET", f"{root}/v1/models", None, timeout))


def load_model(
    base_url: str, key: str, context_length: int = 8192, parallel: int = 1, timeout: float = 600.0
) -> str:
    """Загружает модель, возвращает instance_id (им же адресуется в запросах)."""
    body = {"model": key, "context_length": context_length, "parallel": parallel, "flash_attention": True}
    result = _request("POST", f"{api_root(base_url)}/api/v1/models/load", body, timeout)
    if result.get("status") != "loaded":
        raise LMStudioAdminError(f"Не удалось загрузить {key}: {result}")
    return result.get("instance_id", key)


def unload_model(base_url: str, instance_id: str, timeout: float = 60.0) -> None:
    _request("POST", f"{api_root(base_url)}/api/v1/models/unload", {"instance_id": instance_id}, timeout)


def unload_all(base_url: str) -> list[str]:
    unloaded: list[str] = []
    for model in list_models(base_url):
        for instance in model.loaded_instances:
            unload_model(base_url, instance)
            unloaded.append(instance)
    return unloaded
