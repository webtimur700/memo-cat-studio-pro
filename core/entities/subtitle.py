"""Сущности субтитров с word-level таймкодами (нужны для подсветки текущего
слова — Функция 7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordTiming:
    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    words: list[WordTiming]

    @property
    def start_sec(self) -> float:
        return self.words[0].start_sec if self.words else 0.0

    @property
    def end_sec(self) -> float:
        return self.words[-1].end_sec if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)
