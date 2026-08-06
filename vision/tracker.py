"""Трекинг объектов между кадрами (Функция 5: "AI Tracking").

Реализован упрощённый SORT-подобный трекер: сопоставление детекций между
соседними кадрами по IoU (венгерский алгоритм был бы избыточен для 1-3
объектов в кадре, что типично для видео с животными — greedy-сопоставление
по убыванию IoU даёт тот же результат при таком количестве объектов и
кодируется/тестируется значительно проще).

Трек считается потерянным после `max_missed_frames` кадров без совпадения —
это даёт устойчивость к кратковременным промахам детектора (животное на
секунду загородил хвост/предмет), не создавая новый ID из-за одного кадра.

ЧЕСТНОЕ ОГРАНИЧЕНИЕ (пойманное тестом при разработке): чисто IoU-based
сопоставление без предсказания движения теряет объект, если тот успел
сместиться больше чем на ~half своего размера за время пропущенных кадров —
трекер не экстраполирует траекторию во время пропуска, просто сравнивает с
последней известной позицией. Для быстро движущихся животных при частых
промахах детектора это может создать новый track_id вместо продолжения
старого. Более устойчивое решение (предсказание позиции по скорости, как в
полноценном SORT/Kalman-фильтре) — кандидат на будущее улучшение, здесь
сознательно выбрана более простая и предсказуемая реализация.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.entities.detection import Detection

MIN_IOU_FOR_MATCH = 0.3
DEFAULT_MAX_MISSED_FRAMES = 8


@dataclass(slots=True)
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]

    @property
    def is_active(self) -> bool:
        return self.missed_frames == 0


class ObjectTracker:
    def __init__(
        self,
        min_iou_for_match: float = MIN_IOU_FOR_MATCH,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    ) -> None:
        self._min_iou = min_iou_for_match
        self._max_missed = max_missed_frames
        self._tracks: list[Track] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def update(self, detections: list[Detection]) -> list[Track]:
        """Обновляет треки новыми детекциями одного кадра. Возвращает список
        активных треков (включая только что созданные) после апдейта.
        """
        unmatched_detections = list(detections)
        matched_track_ids: set[int] = set()

        # Greedy-сопоставление: для каждого активного трека ищем детекцию
        # того же класса с максимальным IoU выше порога.
        for track in self._tracks:
            if not unmatched_detections:
                break

            best_match_idx = -1
            best_iou = self._min_iou

            for idx, detection in enumerate(unmatched_detections):
                if detection.class_id != track.last_detection.class_id:
                    continue
                iou = track.last_detection.bbox.iou(detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            if best_match_idx >= 0:
                matched_detection = unmatched_detections.pop(best_match_idx)
                track.detections.append(matched_detection)
                track.missed_frames = 0
                matched_track_ids.add(track.track_id)

        # Треки, для которых не нашлось совпадения в этом кадре.
        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.missed_frames += 1

        # Оставшиеся детекции — новые объекты, создаём под них новые треки.
        for detection in unmatched_detections:
            new_track = Track(track_id=self._next_track_id, detections=[detection])
            self._next_track_id += 1
            self._tracks.append(new_track)

        self._tracks = [t for t in self._tracks if t.missed_frames <= self._max_missed]
        return [t for t in self._tracks if t.is_active]

    def primary_track(self) -> Track | None:
        """Главный объект в кадре для автокадрирования (Функция 5) — трек
        с наибольшим суммарным временем присутствия в кадре, а не просто
        последний по ID: это устойчивее к случайному кратковременному
        объекту, мелькнувшему на пару кадров.
        """
        if not self._tracks:
            return None
        return max(self._tracks, key=lambda t: len(t.detections))
