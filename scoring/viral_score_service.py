"""Формула Viral Score (Функция 3).

В конфиге (config/default_settings.yaml) заложены веса для 5 сигналов — motion_intensity,
scene_change, audio_event, face_prominence, speech_presence. Сейчас считаются motion, scene,
face и (если найден YAMNet, audio/event_classifier.py) audio_event — лай/мяуканье/смех.
Веса нормируются на сумму РЕАЛЬНО использованных сигналов: нет YAMNet или у видео нет звука —
audio_event выпадает из формулы, а не занижает все оценки на его долю. speech_presence пока
не считается (нужен VAD).
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
    audio_event: float | None = None   # 0..1, звуковые события окна; None — классификатор недоступен


def compute_viral_score(inputs: ScoreInputs, weights: ViralScoreSettings) -> int:
    scene_change_score = min(1.0, inputs.scene_change_count / SCENE_CHANGE_SATURATION_COUNT)

    used_weight_sum = (
        weights.weight_motion_intensity + weights.weight_scene_change + weights.weight_face_prominence
    )
    if inputs.audio_event is not None:
        used_weight_sum += weights.weight_audio_event
    if used_weight_sum <= 0:
        return 0

    raw = (
        weights.weight_motion_intensity * inputs.motion_intensity
        + weights.weight_scene_change * scene_change_score
        + weights.weight_face_prominence * inputs.detection_presence
        + weights.weight_audio_event * (inputs.audio_event or 0.0)
    )
    normalized = raw / used_weight_sum
    return max(0, min(100, round(normalized * 100)))
