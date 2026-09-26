"""Один запрос на весь текстовый набор клипа: 10 заголовков, SEO-описание, хештеги (JSON).

Раньше это были три запроса (titles/description/hashtags), и модель трижды заново обрабатывала одни и те же
кадр и расшифровку. Правила стиля — те же, что в отдельных промптах; ответ — JSON по схеме CLIP_CONTENT_SCHEMA
(LM Studio ограничивает генерацию схемой, response_format=json_schema), поэтому сбоев формата нет.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm.prompts.common import grounding_note
from llm.prompts.hashtags_prompt import parse_hashtags

TITLE_COUNT = 10
MAX_HASHTAGS = 30
MAX_DESCRIPTION_CHARS = 500
HASHTAGS_MIN_CHARS = 150    # ~20+ тегов; короче — модель поленилась
HASHTAGS_MAX_CHARS = 700
MAX_TOKENS = 2048   # ~10 заголовков + описание + 30 хештегов — около 500-700 токенов; запас на длинные ответы

SYSTEM_PROMPT = (
    "Ты — контент-мейкер и SEO-специалист YouTube Shorts про животных. По описанию ролика ты готовишь всё "
    "оформление сразу и отвечаешь ТОЛЬКО JSON-объектом, без пояснений и без markdown. "
    "Заголовки и описание пиши на РУССКОМ языке, хештеги — на английском.\n"
    f"titles — ровно {TITLE_COUNT} разных вирусных заголовков на русском: короткие (до 60 символов), цепляющие, "
    "в стиле MrBeast, Daily Dose of Internet, FailArmy и Funny Animals: интрига, эмоция, иногда КАПС для "
    "акцента, эмодзи уместны, но не в каждом заголовке; без хештегов в заголовках.\n"
    "description — SEO-описание на русском: 2-4 предложения, релевантные ключевые слова для поиска, без спама и "
    f"без хештегов внутри текста, не длиннее {MAX_DESCRIPTION_CHARS} символов.\n"
    f"hashtags — одна строка: ровно {MAX_HASHTAGS} релевантных хештегов на английском через пробел, как в "
    "поиске YouTube: смесь широких (#cats, #animals) и узких (#catsoftiktok, #catfails), каждый начинается с #, "
    "без пробелов внутри."
)

CLIP_CONTENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "string"}, "minItems": TITLE_COUNT, "maxItems": TITLE_COUNT},
        "description": {"type": "string"},
        # строкой, а не массивом: без кавычек и запятых на каждый тег это ~90 токенов (~6 с) меньше на клип;
        # длина ограничена, чтобы генерация не зациклилась (свободный список хештегов однажды не остановился)
        "hashtags": {"type": "string", "minLength": HASHTAGS_MIN_CHARS, "maxLength": HASHTAGS_MAX_CHARS},
    },
    "required": ["titles", "description", "hashtags"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class ParsedClipContent:
    titles: tuple[str, ...]
    description: str
    hashtags: tuple[str, ...]


def build_clip_content_prompt(clip_description: str, has_image: bool = False, grounding: bool = True) -> str:
    return (
        f"Вот что происходит в ролике: {clip_description}\n\n"
        f"{grounding_note(has_image, grounding)}"
        "Верни JSON с полями titles, description, hashtags для этого Shorts."
    )


def _extract_json_object(raw: str) -> dict:
    """JSON-объект из ответа; терпимо к обёртке из markdown и пояснений вокруг (если сервер не применил схему)."""
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("в ответе нет JSON-объекта") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"ответ не разобрался как JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("ответ — не JSON-объект")
    return value


def parse_clip_content(raw: str) -> ParsedClipContent:
    """Разбор ответа: чистит значения, но не выдумывает недостающее (пустое поле остаётся пустым — вызывающий
    добирает его отдельным запросом). ValueError — если ответ вообще не JSON-объект."""
    data = _extract_json_object(raw)

    titles_raw = data.get("titles")
    if isinstance(titles_raw, str):
        titles_raw = titles_raw.splitlines()
    titles = tuple(
        t
        for t in (re.sub(r"^\s*\d+\s*[.)\-]\s*", "", str(item)).strip().strip('"').strip() for item in (titles_raw or []))
        if t
    )[:TITLE_COUNT]

    description = str(data.get("description") or "").strip()[:MAX_DESCRIPTION_CHARS]

    tags_raw = data.get("hashtags")
    if isinstance(tags_raw, str):
        tags_raw = tags_raw.split()
    tokens = ["#" + "".join(str(t).split()).lstrip("#") for t in (tags_raw or []) if str(t).strip("# \t")]
    hashtags = tuple(parse_hashtags(" ".join(tokens), max_count=MAX_HASHTAGS))

    return ParsedClipContent(titles, description, hashtags)
