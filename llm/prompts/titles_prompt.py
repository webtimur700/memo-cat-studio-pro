"""Генерация заголовков (Функция 8): 10 вариантов, стиль MrBeast / Daily Dose
of Internet / FailArmy / Funny Animals.
"""

from __future__ import annotations

import re

from core.interfaces.llm_provider import LLMProvider
from llm.prompts.common import complete_with_optional_image, grounding_note

SYSTEM_PROMPT = (
    "Ты — опытный контент-мейкер, который придумывает вирусные заголовки для "
    "YouTube Shorts про животных. Заголовки короткие (до 60 символов), "
    "цепляющие, в стиле каналов MrBeast, Daily Dose of Internet, FailArmy и "
    "Funny Animals: интрига, эмоция, иногда КАПС для акцента, эмодзи уместны, "
    "но не в каждом заголовке. Отвечай СТРОГО нумерованным списком из 10 "
    "пунктов, без вступления и заключения."
)


def build_titles_prompt(clip_description: str, has_image: bool = False, grounding: bool = True) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
        f"{grounding_note(has_image, grounding)}"
        f"Придумай 10 разных вариантов заголовка для этого Shorts."
    )


def parse_titles(raw_response: str, expected_count: int = 10) -> list[str]:
    """Парсит нумерованный список ('1. ...', '1) ...', '1 - ...') из ответа
    модели. Устойчиво к небольшим отклонениям формата (модели не всегда
    идеально следуют инструкции), но не пытается угадывать произвольный текст.
    """
    lines = raw_response.strip().splitlines()
    titles: list[str] = []

    numbered_pattern = re.compile(r"^\s*\d+\s*[.)\-]\s*(.+)$")

    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            title = match.group(1).strip().strip('"').strip()
            if title:
                titles.append(title)

    if not titles:
        # Fallback: ни одна строка не похожа на нумерованный список —
        # берём непустые строки как есть, лучше отдать что-то, чем ничего.
        titles = [line.strip() for line in lines if line.strip()]

    return titles[:expected_count]


def generate_titles(
    provider: LLMProvider,
    clip_description: str,
    count: int = 10,
    max_tokens: int = 4096,
    image_jpeg: bytes | None = None,
    grounding: bool = True,
) -> list[str]:
    prompt = build_titles_prompt(clip_description, has_image=image_jpeg is not None, grounding=grounding)
    raw_response = complete_with_optional_image(provider, SYSTEM_PROMPT, prompt, max_tokens, image_jpeg)
    return parse_titles(raw_response, expected_count=count)
