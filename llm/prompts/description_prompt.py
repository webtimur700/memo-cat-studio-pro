"""Генерация SEO-описания (Функция 9)."""

from __future__ import annotations

from core.interfaces.llm_provider import LLMProvider

SYSTEM_PROMPT = (
    "Ты — SEO-специалист по YouTube. Пиши короткие, ёмкие описания для "
    "Shorts про животных: 2-4 предложения, содержат релевантные ключевые "
    "слова для поиска, без спама и без хештегов внутри текста (хештеги "
    "добавляются отдельно). Отвечай только текстом описания, без пояснений."
)


def build_description_prompt(clip_description: str, max_length_chars: int = 500) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
        f"Напиши SEO-описание для YouTube, не длиннее {max_length_chars} символов."
    )


def generate_description(
    provider: LLMProvider, clip_description: str, max_length_chars: int = 500
) -> str:
    prompt = build_description_prompt(clip_description, max_length_chars)
    raw_response = provider.complete(SYSTEM_PROMPT, prompt, max_tokens=300)
    description = raw_response.strip()
    return description[:max_length_chars]
