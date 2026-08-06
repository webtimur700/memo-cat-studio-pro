"""Пресеты качества экспорта (Функция 19: "качество экспорта"). CRF ниже =
выше качество/больше размер файла — стандартная логика x264.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityPreset:
    crf: int
    x264_preset: str  # компромисс скорость/сжатие кодека, не путать с "quality_preset" пользователя
    audio_bitrate_kbps: int


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "low": QualityPreset(crf=28, x264_preset="veryfast", audio_bitrate_kbps=96),
    "medium": QualityPreset(crf=23, x264_preset="fast", audio_bitrate_kbps=128),
    "high": QualityPreset(crf=18, x264_preset="medium", audio_bitrate_kbps=192),
}


def resolve_quality_preset(name: str) -> QualityPreset:
    return QUALITY_PRESETS.get(name, QUALITY_PRESETS["high"])
