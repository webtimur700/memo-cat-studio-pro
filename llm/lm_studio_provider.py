"""Клиент к локальному серверу LM Studio (OpenAI-совместимый API,
`http://localhost:1234/v1` по умолчанию — см. .env.example).

Реализовано через стандартную библиотеку (`urllib.request`), а не httpx —
это единственный HTTP-вызов во всём проекте, тянуть отдельную зависимость
ради него не оправдано, а urllib позволяет протестировать клиент вживую
против настоящего локального HTTP-сервера прямо в CI/песочнице без сети
наружу (см. tests/integration/test_lm_studio_provider.py).
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from core.exceptions import MemoCatError
from core.interfaces.llm_provider import LLMProvider

# LM Studio — локальный сервер: ходим к нему напрямую, минуя системный прокси.
# urllib не понимает CIDR в no_proxy (например 127.0.0.0/8), из-за чего запросы
# к 127.0.0.1 уходили бы в http_proxy и получали 503.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMRequestError(MemoCatError):
    pass


class LLMUnavailableError(LLMRequestError):
    """Сервер LM Studio недоступен (не запущен / сеть / таймаут)."""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Убирает блок рассуждений <think>…</think> у reasoning-моделей (Qwen3 и т.п.).

    Если блок открыт, но не закрыт — модель упёрлась в max_tokens посреди
    рассуждения, ответа после него нет; возвращаем пустую строку, чтобы вызывающий
    код не принял мысли модели за результат.
    """
    without_closed = _THINK_BLOCK.sub("", text)
    if re.search(r"<think>", without_closed, flags=re.IGNORECASE):
        return ""
    # Некоторые шаблоны отдают только закрывающий </think> (открывающий уже в промпте).
    if "</think>" in without_closed.lower():
        without_closed = re.split(r"</think>", without_closed, flags=re.IGNORECASE)[-1]
    return without_closed.strip()


@dataclass(frozen=True, slots=True)
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    timeout_sec: float = 120.0
    model_override: str | None = None
    # "none" отключает рассуждения (reasoning_effort в /v1/chat/completions), "low"/"medium"
    # ограничивает; None — не передавать параметр (поведение модели по умолчанию).
    reasoning_effort: str | None = None
    context_length: int = 8192        # контекст при загрузке модели самим приложением (по умолчанию LM Studio берёт 262144)
    use_vision: bool = True           # отправлять кадр обложки, если модель умеет vision
    manage_models: bool = True        # приложение само выбирает и загружает модель (llm/managed_provider.py)

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "LMStudioConfig":
        """LM_STUDIO_BASE_URL / LM_STUDIO_REQUEST_TIMEOUT_SEC / LM_STUDIO_MODEL_OVERRIDE
        из .env (если он есть) и переменных окружения; остальное — значения по умолчанию.
        """
        from dotenv import load_dotenv

        load_dotenv(env_path or Path(".env"), override=False)
        defaults = cls()
        return cls(
            base_url=os.environ.get("LM_STUDIO_BASE_URL", "").strip() or defaults.base_url,
            timeout_sec=float(os.environ.get("LM_STUDIO_REQUEST_TIMEOUT_SEC", "") or defaults.timeout_sec),
            model_override=os.environ.get("LM_STUDIO_MODEL_OVERRIDE", "").strip() or None,
            reasoning_effort=os.environ.get("LM_STUDIO_REASONING_EFFORT", "").strip() or None,
            context_length=int(os.environ.get("LM_STUDIO_CONTEXT_LENGTH", "") or defaults.context_length),
            use_vision=os.environ.get("LM_STUDIO_USE_VISION", "1").strip().lower() not in ("0", "false", "no"),
            manage_models=os.environ.get("LM_STUDIO_MANAGE_MODELS", "1").strip().lower() not in ("0", "false", "no"),
        )


class LMStudioProvider(LLMProvider):
    def __init__(self, config: LMStudioConfig | None = None) -> None:
        self._config = config or LMStudioConfig()
        self.last_usage: dict = {}   # usage последнего ответа (токены), для замеров
        self.supports_vision: bool = False   # выставляет фабрика по метаданным выбранной модели
        self.last_model: str = ""

    def list_models(self) -> list[str]:
        url = f"{self._config.base_url}/models"
        try:
            request = urllib.request.Request(url, method="GET")
            with _LOCAL_OPENER.open(request, timeout=self._config.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailableError(
                f"Не удалось получить список моделей LM Studio по {url}: {exc}. "
                f"Убедитесь, что в LM Studio запущен Local Server (вкладка Developer)."
            ) from exc

        return [model["id"] for model in payload.get("data", [])]

    def _resolve_model(self) -> str:
        if self._config.model_override:
            return self._config.model_override

        available = self.list_models()
        if not available:
            raise LLMRequestError(
                "В LM Studio не загружено ни одной модели. Загрузите модель во "
                "вкладке Developer/Local Server перед генерацией контента."
            )
        # LM Studio отдаёт список из фактически загруженных моделей — берём
        # первую, а не гадаем: это ровно та модель, что видна на скриншоте
        # пользователя в LM Studio ("Currently Loaded").
        return available[0]

    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512, images: list[bytes] | None = None
    ) -> str:
        """images — JPEG-кадры (vision-модели): уходят data-URL'ами вместе с текстом."""
        model = self._resolve_model()
        url = f"{self._config.base_url}/chat/completions"

        user_content: object = user_prompt
        if images:
            user_content = [{"type": "text", "text": user_prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}
                for img in images
            ]
        payload_body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }
        if self._config.reasoning_effort:
            payload_body["reasoning_effort"] = self._config.reasoning_effort
        body = json.dumps(payload_body).encode("utf-8")

        request = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )

        try:
            with _LOCAL_OPENER.open(request, timeout=self._config.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailableError(f"Ошибка запроса к LM Studio ({model}): {exc}") from exc

        self.last_usage = payload.get("usage", {}) or {}
        self.last_model = model
        try:
            raw_content = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMRequestError(f"Неожиданный формат ответа LM Studio: {payload}") from exc

        content = strip_reasoning(raw_content)
        if not content:
            raise LLMRequestError(
                f"Пустой ответ модели {model} после удаления <think>: рассуждение не уместилось "
                f"в max_tokens={max_tokens}. Увеличьте лимит или отключите thinking в LM Studio."
            )
        return content
