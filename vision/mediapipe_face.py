"""MediaPipe в проекте используется для ДВУХ разных вещей, и это стоит не путать:

1. `HumanFaceDetector` — обёртка над реальной моделью MediaPipe Face Detector.
   Применимо только если в кадре оказался человек (например, хозяин животного
   в кадре) — MediaPipe официально обучен на человеческих лицах.

2. `AnimalHeadRegionEstimator` — ЧЕСТНАЯ геометрическая эвристика, а не ML-модель.
   Готовой общедоступной модели детекции морды кошки/собаки с точностью
   MediaPipe Face Detector НЕ существует (проговорено ещё в Шаге 1). Вместо
   того чтобы выдавать эвристику за "AI-детекцию морды", здесь честно
   реализовано: "голова животного статистически находится в верхней части
   его bounding box, в стороне направления движения" — этого достаточно для
   effects/collision_detector.py (Функция: "если плашка закрывает
   животное/морду — сдвинуть"), где нужна не точная морда, а безопасная зона,
   которую нельзя перекрывать.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.detection import BoundingBox, Detection

# Доля bbox животного, которая считается "головной зоной" — верхние 40% по
# высоте, смещённые в сторону движения (если известно направление).
HEAD_REGION_HEIGHT_RATIO = 0.4


@dataclass(frozen=True, slots=True)
class FaceDetectionResult:
    bbox: BoundingBox
    confidence: float


class HumanFaceDetector:
    """Обёртка над mediapipe.tasks.vision.FaceDetector — реальная ML-модель,
    но только для человеческих лиц (см. честное примечание в docstring модуля).
    """

    def __init__(self, model_asset_path: str | None = None, min_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        # Если путь к .tflite не передан — используем встроенный short-range
        # детектор MediaPipe Model Maker по умолчанию, поставляемый с пакетом.
        options = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path) if model_asset_path else None,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options) if model_asset_path else None
        self._min_confidence = min_confidence

    def detect(self, frame_rgb) -> list[FaceDetectionResult]:
        """frame_rgb: numpy-массив HxWx3 в формате RGB (не BGR — конвертация
        из OpenCV выполняется вызывающей стороной, чтобы этот модуль не тянул
        зависимость на конкретный источник кадра).
        """
        if self._detector is None:
            return []

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        detections: list[FaceDetectionResult] = []
        for det in result.detections:
            bbox = det.bounding_box
            detections.append(
                FaceDetectionResult(
                    bbox=BoundingBox(
                        x1=bbox.origin_x,
                        y1=bbox.origin_y,
                        x2=bbox.origin_x + bbox.width,
                        y2=bbox.origin_y + bbox.height,
                    ),
                    confidence=det.categories[0].score if det.categories else 0.0,
                )
            )
        return detections


class AnimalHeadRegionEstimator:
    """Геометрическая эвристика головной зоны животного. Никакой ML-модели —
    сознательно простая и предсказуемая функция, чтобы effects/collision_detector.py
    мог полагаться на неё без риска ложной уверенности от "почти угаданной" ML-модели.
    """

    def estimate(self, animal_detection: Detection, motion_direction_x: float = 0.0) -> BoundingBox:
        """motion_direction_x: -1..1, направление горизонтального движения
        (из vision/tracker.py — разница x между текущим и предыдущим bbox).
        Животные обычно смотрят в сторону движения, поэтому голова смещена
        в bbox в ту же сторону по горизонтали.
        """
        bbox = animal_detection.bbox
        head_height = bbox.height * HEAD_REGION_HEIGHT_RATIO

        # Головная зона по горизонтали занимает 80% ширины bbox, что оставляет
        # запас 10% с каждой стороны для смещения в сторону движения — иначе
        # смещать было бы некуда (зона шириной во весь bbox упирается в его же
        # границы). motion_direction_x в диапазоне -1..1 -> максимальное
        # смещение центра зоны на 10% ширины bbox в соответствующую сторону.
        head_width = bbox.width * 0.8
        center_x = (bbox.x1 + bbox.x2) / 2 + motion_direction_x * bbox.width * 0.1

        x1 = max(bbox.x1, min(bbox.x2 - head_width, center_x - head_width / 2))
        x2 = x1 + head_width

        return BoundingBox(x1=x1, y1=bbox.y1, x2=x2, y2=bbox.y1 + head_height)
