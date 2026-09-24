"""Генерация SEO-описания (Функция 9)."""

from __future__ import annotations

from core.interfaces.llm_provider import LLMProvider
from llm.prompts.common import complete_with_optional_image, grounding_note

SYSTEM_PROMPT = (
    "Ты — SEO-специалист по YouTube. Пиши короткие, ёмкие описания для "
    "Shorts про животных: 2-4 предложения, содержат релевантные ключевые "
    "слова для поиска, без спама и без хештегов внутри текста (хештеги "
    "добавляются отдельно). Отвечай только текстом описания, без пояснений."
)


def build_description_prompt(
    clip_description: str, max_length_chars: int = 500, has_image: bool = False, grounding: bool = True
) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
        f"{grounding_note(has_image, grounding)}"
        f"Напиши SEO-описание для YouTube, не длиннее {max_length_chars} символов."
    )


def generate_description(
    provider: LLMProvider,
    clip_description: str,
    max_length_chars: int = 500,
    max_tokens: int = 3072,
    image_jpeg: bytes | None = None,
    grounding: bool = True,
) -> str:
    prompt = build_description_prompt(
        clip_description, max_length_chars, has_image=image_jpeg is not None, grounding=grounding
    )
    raw_response = complete_with_optional_image(provider, SYSTEM_PROMPT, prompt, max_tokens, image_jpeg)
    description = raw_response.strip()
    return description[:max_length_chars]
