"""Перевод субтитров через LLM с сохранением тайминга.

Whisper даёт слова с таймкодами на языке оригинала (например, китайская речь в
ролике для русскоязычного канала). Слова склеиваются в реплики (по паузам), реплики
переводятся одним запросом к LLM, а слова перевода раскладываются по времени
реплики пропорционально длине — подсветка текущего слова остаётся работоспособной.
"""

from __future__ import annotations

import re

from loguru import logger

from core.entities.subtitle import WordTiming
from core.interfaces.llm_provider import LLMProvider

LANGUAGE_NAMES = {"ru": "русский", "en": "английский", "uk": "украинский", "de": "немецкий", "es": "испанский"}

MAX_GAP_SEC = 0.7           # пауза длиннее — новая реплика
MAX_UNIT_SEC = 6.0
MAX_UNIT_WORDS = 14
MAX_TOKENS = 3000

SYSTEM_PROMPT = (
    "Ты переводчик субтитров к коротким видео с животными. Переводи реплики точно и естественно. "
    "Междометия, смех и неразборчивые звуки передавай короткими междометиями (\"ха-ха\", \"ой\", \"ай\"). "
    "Не добавляй ничего от себя и не объясняй. Отвечай СТРОГО нумерованным списком: "
    "столько строк, сколько реплик, в том же порядке, формат \"номер. перевод\"."
)


def _is_cjk(ch: str) -> bool:
    return "぀" <= ch <= "ヿ" or "㐀" <= ch <= "鿿" or "가" <= ch <= "힯"


def _join_words(words: list[WordTiming]) -> str:
    """Китайские/японские слова склеиваются без пробелов, остальные — через пробел."""
    text = ""
    for word in words:
        if text and not (_is_cjk(text[-1]) and word.text and _is_cjk(word.text[0])):
            text += " "
        text += word.text
    return text.strip()


def split_into_units(words: list[WordTiming]) -> list[list[WordTiming]]:
    units: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for word in words:
        if current:
            gap = word.start_sec - current[-1].end_sec
            too_long = word.end_sec - current[0].start_sec > MAX_UNIT_SEC or len(current) >= MAX_UNIT_WORDS
            if gap > MAX_GAP_SEC or too_long:
                units.append(current)
                current = []
        current.append(word)
    if current:
        units.append(current)
    return units


def parse_numbered(raw: str, expected: int) -> list[str] | None:
    """Разбор "1. текст" в список; None, если строк не ровно expected."""
    found: dict[int, str] = {}
    for line in raw.splitlines():
        match = re.match(r"^\s*(\d+)\s*[.)\-:]\s*(.+?)\s*$", line)
        if match:
            found.setdefault(int(match.group(1)), match.group(2).strip().strip('"«»'))
    if sorted(found) != list(range(1, expected + 1)):
        return None
    return [found[i] for i in range(1, expected + 1)]


def spread_words(text: str, start: float, end: float) -> list[WordTiming]:
    """Раскладывает слова перевода по интервалу реплики пропорционально длине слов."""
    tokens = text.split()
    if not tokens:
        return []
    weights = [max(1, len(t)) for t in tokens]
    total = sum(weights)
    span = max(end - start, 0.05 * len(tokens))
    result: list[WordTiming] = []
    cursor = start
    for token, weight in zip(tokens, weights):
        step = span * weight / total
        result.append(WordTiming(token, cursor, cursor + step))
        cursor += step
    return result


def translate_words(
    provider: LLMProvider, words: list[WordTiming], target_language: str = "ru"
) -> list[WordTiming] | None:
    """Слова на языке перевода с таймингами или None, если перевод не удался
    (тогда вызывающий оставляет оригинал)."""
    if not words:
        return []
    units = split_into_units(words)
    numbered = "\n".join(f"{i}. {_join_words(unit)}" for i, unit in enumerate(units, 1))
    language_name = LANGUAGE_NAMES.get(target_language, target_language)
    prompt = f"Переведи реплики на {language_name}:\n\n{numbered}"
    try:
        raw = provider.complete(SYSTEM_PROMPT, prompt, max_tokens=MAX_TOKENS)
    except Exception as exc:
        logger.warning("Перевод субтитров не удался: {} — оставляю язык оригинала", exc)
        return None

    translations = parse_numbered(raw, len(units))
    if translations is None:
        logger.warning("Перевод субтитров: ответ LLM не соответствует числу реплик — оставляю оригинал")
        return None

    translated: list[WordTiming] = []
    for unit, text in zip(units, translations):
        translated.extend(spread_words(text, unit[0].start_sec, unit[-1].end_sec))
    return translated
