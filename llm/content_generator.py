"""Генерация контента клипа через LLM: 10 заголовков, SEO-описание, до 30 хештегов.

Описание для модели — текст транскрипции момента (то, что реально говорится в
ролике). Основной путь — ОДИН запрос с ответом по JSON-схеме (llm/prompts/clip_content_prompt.py):
модель обрабатывает кадр и расшифровку один раз. Чего в ответе не хватило (или если провайдер/сервер
схемы не поддерживает), добирается отдельными запросами — заголовки, описание, хештеги независимы:
сбой одного не отменяет остальные. Недоступность сервера (LM Studio не запущена) прекращает попытки
сразу, чтобы не ждать таймаут на каждый запрос каждого момента.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger

from core.interfaces.llm_provider import LLMProvider
from llm.lm_studio_provider import LLMRequestError, LLMUnavailableError
from llm.prompts.clip_content_prompt import (
    CLIP_CONTENT_SCHEMA,
    MAX_TOKENS as COMBINED_MAX_TOKENS,
    SYSTEM_PROMPT as COMBINED_SYSTEM_PROMPT,
    build_clip_content_prompt,
    parse_clip_content,
)
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
    used_image: bool = False          # в запросах реально участвовал кадр
    requests: int = 0                 # сколько запросов к модели ушло на этот клип

    @property
    def is_empty(self) -> bool:
        return not (self.titles or self.description or self.hashtags)


def build_clip_description(transcript: str) -> str:
    text = " ".join(transcript.split())[:MAX_TRANSCRIPT_CHARS]
    if not text:
        return NO_SPEECH_DESCRIPTION
    return f"Забавный момент с животными. Что говорят в ролике: «{text}»"


class _Run:
    """Состояние генерации одного клипа: кадр (отбрасывается, если модель его не приняла), счётчик запросов, ошибки."""

    def __init__(self, image: bytes | None) -> None:
        self.image = image
        self.image_worked = False
        self.requests = 0
        self.errors: list[str] = []
        self.unavailable = False

    def call(self, name: str, step):
        """step(image) -> результат или None при сбое (причина в errors). Кадр не принят — повтор без кадра."""
        try:
            try:
                self.requests += 1
                result = step(self.image)
                self.image_worked = self.image_worked or self.image is not None
                return result
            except LLMUnavailableError:
                raise
            except LLMRequestError as exc:
                if self.image is None:
                    raise
                logger.warning("LLM: кадр не принят ({}) — повторяю {} в текстовом режиме", exc, name)
                self.image = None
                self.requests += 1
                return step(None)
        except LLMUnavailableError as exc:
            self.errors.append(f"{name}: {exc}")
            self.unavailable = True   # сервер недоступен — остальные запросы бессмысленны
            logger.warning("LLM недоступна: {} — заголовок по умолчанию", exc)
        except Exception as exc:
            self.errors.append(f"{name}: {exc}")
            logger.warning("LLM: {} не сгенерированы: {}", name, exc)
        return None


def _generate_combined(provider, description: str, run: _Run):
    """Один запрос -> ParsedClipContent или None (тогда всё делается по частям)."""

    def step(image: bytes | None):
        raw = provider.complete_json(
            COMBINED_SYSTEM_PROMPT,
            build_clip_content_prompt(description, has_image=image is not None),
            CLIP_CONTENT_SCHEMA,
            max_tokens=COMBINED_MAX_TOKENS,
            images=None if image is None else [image],
            schema_name="clip_content",
        )
        return parse_clip_content(raw)

    errors_before = len(run.errors)
    parsed = run.call("заголовки, описание и хештеги", step)
    if parsed is None and not run.unavailable:
        # причина уже в логе; отдельные запросы могут получиться, поэтому не считаем это ошибкой клипа
        del run.errors[errors_before:]
        logger.info("LLM: единый запрос не удался — делаю заголовки, описание и хештеги по отдельности")
    return parsed


def generate_clip_content(
    provider: LLMProvider, transcript: str, image_jpeg: bytes | None = None, combined: bool = True
) -> ClipContent:
    """image_jpeg — кадр момента для vision-моделей. Если модель кадр не приняла
    (не vision / ошибка запроса), шаг повторяется в текстовом режиме, а последующие
    шаги уже идут без кадра. combined=False — три отдельных запроса (как раньше)."""
    started = time.perf_counter()
    description_for_llm = build_clip_description(transcript)
    run = _Run(image_jpeg)
    results: dict[str, object] = {}

    if combined and callable(getattr(provider, "complete_json", None)):
        parsed = _generate_combined(provider, description_for_llm, run)
        if parsed is not None:
            results = {"заголовки": parsed.titles, "описание": parsed.description, "хештеги": parsed.hashtags}

    steps = (
        ("заголовки", lambda img: generate_titles(provider, description_for_llm, count=TITLE_COUNT, image_jpeg=img)),
        ("описание", lambda img: generate_description(provider, description_for_llm, image_jpeg=img)),
        ("хештеги", lambda img: generate_hashtags(provider, description_for_llm, max_count=MAX_HASHTAGS, image_jpeg=img)),
    )
    for name, step in steps:
        if run.unavailable:
            break
        if results.get(name):
            continue   # уже получено единым запросом
        value = run.call(name, step)
        if value is not None:
            results[name] = value

    logger.info(
        "LLM: тексты клипа готовы за {:.1f} с, запросов: {}{}",
        time.perf_counter() - started, run.requests, " (кадр учтён)" if run.image_worked else "",
    )
    return ClipContent(
        titles=tuple(results.get("заголовки", ())),  # type: ignore[arg-type]
        description=str(results.get("описание", "")),
        hashtags=tuple(results.get("хештеги", ())),  # type: ignore[arg-type]
        errors=tuple(run.errors),
        used_image=run.image_worked,
        requests=run.requests,
    )
