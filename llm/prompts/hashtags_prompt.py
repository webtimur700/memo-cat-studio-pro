"""Генерация хештегов (Функция 10): до 30 штук."""

from __future__ import annotations

import re

from core.interfaces.llm_provider import LLMProvider

SYSTEM_PROMPT = (
    "Ты — SEO-специалист по YouTube Shorts про животных. Генерируешь "
    "релевантные хештеги: смесь широких (#cats, #animals) и узких "
    "(#catsoftiktok, #catfails) тегов. Отвечай СТРОГО списком хештегов через "
    "пробел, каждый начинается с #, без пояснений и нумерации."
)


def build_hashtags_prompt(clip_description: str, max_count: int = 30) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
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


def generate_hashtags(provider: LLMProvider, clip_description: str, max_count: int = 30) -> list[str]:
    prompt = build_hashtags_prompt(clip_description, max_count)
    raw_response = provider.complete(SYSTEM_PROMPT, prompt, max_tokens=300)
    return parse_hashtags(raw_response, max_count=max_count)
