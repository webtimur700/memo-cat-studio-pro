"""Контракт LLM-провайдера. llm/prompts/*.py и все сервисы генерации
контента (заголовки/описание/хештеги) зависят только от этого протокола —
конкретный провайдер (LM Studio/Ollama) подставляется через DI и может быть
заменён без изменения кода, который генерирует контент.
"""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512, images: list[bytes] | None = None
    ) -> str:
        """Возвращает текстовый ответ модели на промпт (images — JPEG-кадры для vision-моделей)."""
        ...

    def list_models(self) -> list[str]:
        """Возвращает ID моделей, доступных прямо сейчас через этот провайдер."""
        ...
