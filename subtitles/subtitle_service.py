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

    def _ensure_model_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Загрузка faster-whisper: model={}, device={}, compute_type={}",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = "ru") -> list[WordTiming]:
        if not audio_path.exists():
            raise TranscriptionError(f"Аудиофайл не найден: {audio_path}")

        model = self._ensure_model_loaded()

        try:
            segments, info = model.transcribe(
                str(audio_path), word_timestamps=True, language=language
            )
        except Exception as exc:  # faster-whisper/ctranslate2 могут бросать разные исключения
            raise TranscriptionError(f"Ошибка транскрипции {audio_path}: {exc}") from exc

        words: list[WordTiming] = []
        for segment in segments:
            if segment.words is None:
                continue
            for word in segment.words:
                words.append(
                    WordTiming(text=word.word.strip(), start_sec=word.start, end_sec=word.end)
                )

        logger.info("Транскрипция завершена: {} слов, язык={}", len(words), getattr(info, "language", "?"))
        return words

    def transcribe_to_segments(
        self,
        audio_path: Path,
        language: str | None = "ru",
        max_words_per_segment: int = DEFAULT_MAX_WORDS_PER_SEGMENT,
        max_segment_duration_sec: float = DEFAULT_MAX_SEGMENT_DURATION_SEC,
    ) -> list[SubtitleSegment]:
        words = self.transcribe(audio_path, language)
        return group_words_into_segments(words, max_words_per_segment, max_segment_duration_sec)
