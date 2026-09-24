"""Генерация хештегов (Функция 10): до 30 штук."""

from __future__ import annotations

import re

from core.interfaces.llm_provider import LLMProvider
from llm.prompts.common import complete_with_optional_image, grounding_note

SYSTEM_PROMPT = (
    "Ты — SEO-специалист по YouTube Shorts про животных. Генерируешь "
    "релевантные хештеги: смесь широких (#cats, #animals) и узких "
    "(#catsoftiktok, #catfails) тегов. Отвечай СТРОГО списком хештегов через "
    "пробел, каждый начинается с #, без пояснений и нумерации."
)


def build_hashtags_prompt(
    clip_description: str, max_count: int = 30, has_image: bool = False, grounding: bool = True
) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
        f"{grounding_note(has_image, grounding)}"
        f"Дай до {max_count} хештегов для этого Shorts."
    )


def parse_hashtags(raw_response: str, max_count: int = 30) -> list[str]:
    found = re.findall(r"#[\w\d_]+", raw_response, flags=re.UNICODE)
    # Убираем дубликаты, сохраняя порядок (важно для релевантности — первые
    # теги обычно самые релевантные, порядок терять нельзя).
    seen: set[str] = set()
    unique: list[str] = []
    for tag in found:
        normalized = tag.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(tag)
    return unique[:max_count]


def generate_hashtags(
    provider: LLMProvider,
    clip_description: str,
    max_count: int = 30,
    max_tokens: int = 3072,
    image_jpeg: bytes | None = None,
    grounding: bool = True,
) -> list[str]:
    prompt = build_hashtags_prompt(clip_description, max_count, has_image=image_jpeg is not None, grounding=grounding)
    raw_response = complete_with_optional_image(provider, SYSTEM_PROMPT, prompt, max_tokens, image_jpeg)
    return parse_hashtags(raw_response, max_count=max_count)
