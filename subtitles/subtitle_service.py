"""Оркестрация транскрипции (Функция 7).

Транскрипция выполняется faster-whisper с `word_timestamps=True` — это даёт
именно то, что нужно для подсветки текущего слова, а не только границы фраз.

ЧЕСТНОЕ ПРИМЕЧАНИЕ: faster-whisper не установлен в этой песочнице (нет сети),
поэтому WhisperTranscriber.transcribe() не протестирован вживую. Написан по
официальному публичному API faster-whisper (`WhisperModel.transcribe(...,
word_timestamps=True)`, `segment.words[i].word/.start/.end`). Чистая логика
группировки слов в отображаемые сегменты (`group_words_into_segments`) не
зависит от faster-whisper и протестирована на синтетических данных отдельно.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.entities.subtitle import SubtitleSegment, WordTiming
from core.exceptions import MemoCatError

DEFAULT_MAX_WORDS_PER_SEGMENT = 4
DEFAULT_MAX_SEGMENT_DURATION_SEC = 2.5

# Whisper на музыке/шуме/смехе выдаёт заученные фразы из обучающих субтитров. Отсекаем их
# и сегменты, в которых модель сама не уверена (по её же оценкам).
MIN_SEGMENT_AVG_LOGPROB = -1.5
NO_SPEECH_PROB_LIMIT = 0.6
NO_SPEECH_LOGPROB_LIMIT = -1.0
HALLUCINATION_MARKERS = (
    "субтитры сделал", "субтитры создавал", "субтитры подготовил", "редактор субтитров",
    "корректор", "продолжение следует", "спасибо за просмотр", "подписывайтесь на канал",
    "thanks for watching", "thank you for watching", "subtitles by", "amara.org",
    "字幕", "请不吝点赞", "订阅", "轉載", "点赞",
)


def is_hallucination(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in HALLUCINATION_MARKERS)


def should_drop_segment(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    """Сегмент — вероятная галлюцинация/мусор: известная "заученная" фраза, очень низкая
    уверенность модели, либо "нет речи" при невысокой уверенности."""
    if is_hallucination(text):
        return True
    if avg_logprob < MIN_SEGMENT_AVG_LOGPROB:
        return True
    return no_speech_prob > NO_SPEECH_PROB_LIMIT and avg_logprob < NO_SPEECH_LOGPROB_LIMIT


class TranscriptionError(MemoCatError):
    pass


def group_words_into_segments(
    words: list[WordTiming],
    max_words_per_segment: int = DEFAULT_MAX_WORDS_PER_SEGMENT,
    max_segment_duration_sec: float = DEFAULT_MAX_SEGMENT_DURATION_SEC,
) -> list[SubtitleSegment]:
    """Группирует плоский список слов с таймкодами в короткие сегменты для
    показа на экране (короткие Shorts-субтитры — 2-4 слова на экране,
    а не длинные строки как в обычном кино).

    Разрыв сегмента происходит по любому из условий: набралось
    max_words_per_segment слов, ИЛИ сегмент уже длится дольше
    max_segment_duration_sec, ИЛИ между словами пауза больше 0.6с (конец
    фразы/вдох — разумная точка для нового сегмента).
    """
    if not words:
        return []

    segments: list[SubtitleSegment] = []
    current_words: list[WordTiming] = [words[0]]

    for prev_word, word in zip(words, words[1:]):
        gap = word.start_sec - prev_word.end_sec
        current_duration = word.end_sec - current_words[0].start_sec

        should_break = (
            len(current_words) >= max_words_per_segment
            or current_duration > max_segment_duration_sec
            or gap > 0.6
        )

        if should_break:
            segments.append(SubtitleSegment(words=current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        segments.append(SubtitleSegment(words=current_words))

    return segments


class WhisperTranscriber:
    def __init__(self, model_size: str = "small", compute_type: str = "int8", device: str = "cpu") -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._device = device
        self._model = None  # ленивая инициализация — модель грузится в память только при первом вызове
        self.last_language: str | None = None   # язык, определённый в последней транскрипции
        self.last_language_probability: float = 0.0

    def _ensure_model_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Загрузка faster-whisper: model={}, device={}, compute_type={}",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = self._load_model(WhisperModel)
        return self._model

    def _load_model(self, model_cls):
        """Сначала только локальный кэш (~/.cache/huggingface): без обращения к
        сети, поэтому не зависит от прокси в окружении (ALL_PROXY=socks:// ломает
        httpx с "Unknown scheme for proxy URL"). Сеть — только если модели в
        кэше нет.
        """
        try:
            return model_cls(
                self._model_size, device=self._device, compute_type=self._compute_type,
                local_files_only=True,
            )
        except Exception as cache_exc:
            logger.info(
                "Модель {} не найдена в локальном кэше ({}) — пробую скачать",
                self._model_size, type(cache_exc).__name__,
            )
        from core.proxy_env import make_httpx_safe

        for change in make_httpx_safe():   # socks:// в ALL_PROXY роняет httpx (huggingface_hub) до запроса
            logger.info("Прокси для загрузки модели: {}", change)
        return model_cls(self._model_size, device=self._device, compute_type=self._compute_type)

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[WordTiming]:
        """language=None (или "auto") — определить язык по звуку. Принудительный язык
        на речи другого языка даёт фонетический мусор (китайская речь с language="ru"
        превращается в "Жанин, уй-ой-ой": уверенность модели падает, время растёт)."""
        if language == "auto":
            language = None
        if not audio_path.exists():
            raise TranscriptionError(f"Аудиофайл не найден: {audio_path}")

        model = self._ensure_model_loaded()

        try:
            segments, info = model.transcribe(
                str(audio_path), word_timestamps=True, language=language,
                vad_filter=True,  # пропускает музыку/тишину: быстрее и без галлюцинаций
            )
        except ValueError as exc:
            # faster-whisper 1.0.x: если после VAD не осталось речи, автоопределение языка падает
            # на max() пустого списка вероятностей — это "речи нет", а не ошибка
            if "max()" in str(exc) and "empty" in str(exc):
                self.last_language = None
                self.last_language_probability = 0.0
                logger.debug("Речи в {} нет (VAD оставил пустое аудио)", audio_path.name)
                return []
            raise TranscriptionError(f"Ошибка транскрипции {audio_path}: {exc}") from exc
        except Exception as exc:  # faster-whisper/ctranslate2 могут бросать разные исключения
            raise TranscriptionError(f"Ошибка транскрипции {audio_path}: {exc}") from exc

        self.last_language = getattr(info, "language", None)
        self.last_language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        words: list[WordTiming] = []
        dropped = 0
        for segment in segments:
            if segment.words is None:
                continue
            if should_drop_segment(
                segment.text, getattr(segment, "avg_logprob", 0.0), getattr(segment, "no_speech_prob", 0.0)
            ):
                dropped += 1
                continue
            for word in segment.words:
                words.append(
                    WordTiming(text=word.word.strip(), start_sec=word.start, end_sec=word.end)
                )

        logger.info(
            "Транскрипция завершена: {} слов, язык={} ({:.0%}), отброшено сегментов-галлюцинаций: {}",
            len(words), self.last_language or "?", self.last_language_probability, dropped,
        )
        return words

    def transcribe_to_segments(
        self,
        audio_path: Path,
        language: str | None = None,
        max_words_per_segment: int = DEFAULT_MAX_WORDS_PER_SEGMENT,
        max_segment_duration_sec: float = DEFAULT_MAX_SEGMENT_DURATION_SEC,
    ) -> list[SubtitleSegment]:
        words = self.transcribe(audio_path, language)
        return group_words_into_segments(words, max_words_per_segment, max_segment_duration_sec)
