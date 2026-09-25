"""Формула Viral Score (Функция 3).

В конфиге (config/default_settings.yaml) заложены веса для сигналов — motion_intensity, scene_change,
audio_event, face_prominence, speech_presence, motion_events. Считаются: motion, scene, face (животное в кадре),
audio_event (YAMNet: лай/мяуканье/смех, audio/event_classifier.py) и motion_events (прыжки, падения, рывки по трекам
YOLO, vision/pose_motion_analyzer.py). speech_presence пока не считается (нужен VAD).

Веса нормируются на сумму РЕАЛЬНО использованных сигналов: нет YAMNet или у видео нет звука — audio_event
выпадает из формулы, а не занижает все оценки на его долю.

motion_events — БОНУС, а не член нормировки: плотный анализ движения (6 кадров/с) дорог и делается только для
лучших окон, поэтому у остальных его нет. Если бы вес входил в знаменатель, окно без прыжков, но с проверкой,
теряло бы очки относительно непроверенных. Бонус только добавляет: score = (сумма вкладов + w·v) / сумма весов.

compute_score_breakdown возвращает не только число, но и вклад каждого сигнала в очках — по нему интерфейс
объясняет, почему выбран момент.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.score import ScoreBreakdown, SignalPart
from core.entities.settings import ViralScoreSettings

SCENE_CHANGE_SATURATION_COUNT = 2

SIGNAL_LABELS = {
    "motion": "Движение",
    "presence": "Животное в кадре",
    "scene": "Смены сцен",
    "audio": "Звуковые события",
    "motion_events": "Прыжки и падения",
}


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    motion_intensity: float       # 0..1
    detection_presence: float     # 0..1, доля кадров окна с детекцией
    scene_change_count: int
    audio_event: float | None = None   # 0..1, звуковые события окна; None — классификатор недоступен
    motion_events: float = 0.0         # 0..1, прыжки/падения/рывки окна (бонус); 0 — не найдены или не проверялось
    audio_detail: str = ""             # какие именно звуки («лай, смех») — для показа пользователю
    motion_events_detail: str = ""     # «2 прыжка, падение»


def compute_score_breakdown(inputs: ScoreInputs, weights: ViralScoreSettings) -> ScoreBreakdown:
    scene_change_score = min(1.0, inputs.scene_change_count / SCENE_CHANGE_SATURATION_COUNT)
    signals: list[tuple[str, float, float, str, bool]] = [
        ("motion", inputs.motion_intensity, weights.weight_motion_intensity, "", False),
        ("presence", inputs.detection_presence, weights.weight_face_prominence, "", False),
        ("scene", scene_change_score, weights.weight_scene_change,
         f"{inputs.scene_change_count} смен(ы)" if inputs.scene_change_count else "", False),
    ]
    if inputs.audio_event is not None:
        signals.append(("audio", inputs.audio_event, weights.weight_audio_event, inputs.audio_detail, False))
    signals.append(("motion_events", inputs.motion_events, weights.weight_motion_events, inputs.motion_events_detail, True))

    used_weight_sum = sum(w for _, _, w, _, bonus in signals if not bonus)
    if used_weight_sum <= 0:
        return ScoreBreakdown(0, ())
    parts = tuple(
        SignalPart(key, SIGNAL_LABELS[key], value, weight, weight * value / used_weight_sum * 100, detail, bonus)
        for key, value, weight, detail, bonus in signals
    )
    total = sum(p.points for p in parts)
    return ScoreBreakdown(max(0, min(100, round(total))), parts)


def compute_viral_score(inputs: ScoreInputs, weights: ViralScoreSettings) -> int:
    return compute_score_breakdown(inputs, weights).score
