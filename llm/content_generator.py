"""Генерация контента клипа через LLM: 10 заголовков, SEO-описание, до 30 хештегов.

Описание для модели — текст транскрипции момента (то, что реально говорится в
ролике). Каждый из трёх запросов независим: сбой одного не отменяет остальные, а
недоступность сервера (LM Studio не запущена) прекращает попытки сразу, чтобы не
ждать таймаут три раза на каждый момент.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from core.interfaces.llm_provider import LLMProvider
from llm.lm_studio_provider import LLMRequestError, LLMUnavailableError
from llm.prompts.description_prompt import generate_description
from llm.prompts.hashtags_prompt import generate_hashtags
from llm.prompts.titles_prompt import generate_titles

TITLE_COUNT = 10
MAX_HASHTAGS = 30
MAX_TRANSCRIPT_CHARS = 2000
NO_SPEECH_DESCRIPTION = "Забавный момент с животными; в ролике нет речи, только видеоряд и звук."


@dataclass(frozen=True, slots=True)
class ClipContent:
    titles: tuple[str, ...] = ()
    description: str = ""
    hashtags: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not (self.titles or self.description or self.hashtags)


def build_clip_description(transcript: str) -> str:
    text = " ".join(transcript.split())[:MAX_TRANSCRIPT_CHARS]
    if not text:
        return NO_SPEECH_DESCRIPTION
    return f"Забавный момент с животными. Что говорят в ролике: «{text}»"


def generate_clip_content(provider: LLMProvider, transcript: str) -> ClipContent:
    description_for_llm = build_clip_description(transcript)
    titles: tuple[str, ...] = ()
    description = ""
    hashtags: tuple[str, ...] = ()
    errors: list[str] = []

    steps = (
        ("заголовки", lambda: generate_titles(provider, description_for_llm, count=TITLE_COUNT)),
        ("описание", lambda: generate_description(provider, description_for_llm)),
        ("хештеги", lambda: generate_hashtags(provider, description_for_llm, max_count=MAX_HASHTAGS)),
    )
    results: dict[str, object] = {}
    for name, step in steps:
        try:
            results[name] = step()
        except LLMUnavailableError as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("LLM недоступна: {} — заголовок по умолчанию", exc)
            break  # сервер недоступен — остальные запросы бессмысленны
        except LLMRequestError as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("LLM: {} не сгенерированы: {}", name, exc)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("LLM: {} не сгенерированы: {}", name, exc)

    titles = tuple(results.get("заголовки", ()))  # type: ignore[arg-type]
    description = str(results.get("описание", ""))
    hashtags = tuple(results.get("хештеги", ()))  # type: ignore[arg-type]
    return ClipContent(titles=titles, description=description, hashtags=hashtags, errors=tuple(errors))
