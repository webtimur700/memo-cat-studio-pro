"""Формула Viral Score (Функция 3).

ЧЕСТНО: в конфиге (config/default_settings.yaml) заложены веса для 5
сигналов — motion_intensity, scene_change, audio_event, face_prominence,
speech_presence. Модуль audio/ (детекция лая/мяу/смеха, VAD) в проекте пока
не реализован — это отдельный кусок работы. Чтобы формула не занижала все
оценки из-за отсутствующих ~40% веса (audio_event+speech_presence), она
перенормируется на сумму РЕАЛЬНО используемых весов (motion+scene+face) —
то есть Viral Score пока отражает только визуальную часть "интересности",
без звука. Когда audio/ появится — добавится audio_score без изменения
сигнатуры вызова.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.settings import ViralScoreSettings

SCENE_CHANGE_SATURATION_COUNT = 2


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    motion_intensity: float       # 0..1
    detection_presence: float     # 0..1, доля кадров окна с детекцией
    scene_change_count: int


def compute_viral_score(inputs: ScoreInputs, weights: ViralScoreSettings) -> int:
    scene_change_score = min(1.0, inputs.scene_change_count / SCENE_CHANGE_SATURATION_COUNT)

    used_weight_sum = (
        weights.weight_motion_intensity + weights.weight_scene_change + weights.weight_face_prominence
    )
    if used_weight_sum <= 0:
        return 0

    raw = (
        weights.weight_motion_intensity * inputs.motion_intensity
        + weights.weight_scene_change * scene_change_score
        + weights.weight_face_prominence * inputs.detection_presence
    )
    normalized = raw / used_weight_sum
    return max(0, min(100, round(normalized * 100)))
