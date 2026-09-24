"""Общие куски промптов: защита от выдуманных деталей и отправка кадра."""

from __future__ import annotations

from core.interfaces.llm_provider import LLMProvider

GROUNDING_TEXT_ONLY = (
    "Если речи в ролике нет или она непонятна — не придумывай конкретных деталей (породу, действия, предметов): пиши общими словами про животных."
)
GROUNDING_WITH_IMAGE = (
    "К сообщению приложен кадр из ролика. Опирайся на то, что реально видно на кадре (какое животное, что оно делает) и на речь. Не выдумывай деталей, которых нет ни на кадре, ни в речи."
)


def grounding_note(has_image: bool, grounding: bool = True) -> str:
    """Строка-инструкция в промпт: с кадром — опираться на кадр, без него — не выдумывать детали.
    grounding=False возвращает "" (исходные промпты — для сравнения в бенчмарке)."""
    if not grounding:
        return ""
    return (GROUNDING_WITH_IMAGE if has_image else GROUNDING_TEXT_ONLY) + "\n\n"


def complete_with_optional_image(
    provider: LLMProvider, system_prompt: str, prompt: str, max_tokens: int, image_jpeg: bytes | None
) -> str:
    if image_jpeg is None:
        return provider.complete(system_prompt, prompt, max_tokens=max_tokens)
    return provider.complete(system_prompt, prompt, max_tokens=max_tokens, images=[image_jpeg])
