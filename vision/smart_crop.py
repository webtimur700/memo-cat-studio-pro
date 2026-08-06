"""Автокадрирование в 1080x1920 со слежением за объектом (Функция 5) и
AI Zoom (Функция 6).

Алгоритм на каждый кадр:
  1. Взять текущий bbox главного трека (vision/tracker.py primary_track()).
  2. Экспоненциально сгладить координаты центра кропа (чтобы камера не
     дёргалась на каждый мелкий шум детектора — рывки хуже, чем небольшая
     задержка следования).
  3. Если объект занимает меньше `min_subject_area_ratio` кадра — увеличить
     zoom_factor (Функция: "если животное маленькое — приближать").
  4. Вычислить прямоугольник кропа нужного аспект-рейшо (9:16) вокруг
     сглаженного центра, с клэмпом в границы оригинального кадра.

Результат — последовательность CropWindow, которую export/-слой (Шаг 10)
использует как ffmpeg crop-фильтр покадрово (через `sendcmd`/purpose-built
инструмент) либо effects-слой рендерит через OpenCV warpAffine/resize.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.detection import BoundingBox, Detection


@dataclass(frozen=True, slots=True)
class CropWindow:
    x1: float
    y1: float
    x2: float
    y2: float
    zoom_factor: float
    timestamp_sec: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class SmartCropPlanner:
    def __init__(
        self,
        source_width: int,
        source_height: int,
        target_aspect_ratio: float = 9 / 16,
        smoothing_alpha: float = 0.15,
        min_subject_area_ratio: float = 0.08,
        max_zoom_factor: float = 2.2,
    ) -> None:
        self._source_width = source_width
        self._source_height = source_height
        self._target_aspect_ratio = target_aspect_ratio
        self._smoothing_alpha = smoothing_alpha
        self._min_subject_area_ratio = min_subject_area_ratio
        self._max_zoom_factor = max_zoom_factor

        self._smoothed_center_x: float | None = None
        self._smoothed_center_y: float | None = None
        self._smoothed_zoom: float = 1.0

    def plan_frame(self, detection: Detection | None) -> CropWindow:
        """Вызывается покадрово (или на сэмплированной частоте с интерполяцией
        между вызовами — это делает export-слой). Если detection is None
        (объект временно потерян трекером) — кроп продолжает плавно двигаться
        к последней известной позиции, а не дёргается в центр кадра.
        """
        if detection is not None:
            target_center_x, target_center_y = detection.bbox.center
            target_zoom = self._compute_zoom_factor(detection.bbox)
        elif self._smoothed_center_x is not None:
            target_center_x, target_center_y = self._smoothed_center_x, self._smoothed_center_y
            target_zoom = self._smoothed_zoom
        else:
            # Ни одной детекции ещё не было — кадрируем по центру оригинала.
            target_center_x = self._source_width / 2
            target_center_y = self._source_height / 2
            target_zoom = 1.0

        self._smoothed_center_x = self._ema(self._smoothed_center_x, target_center_x)
        self._smoothed_center_y = self._ema(self._smoothed_center_y, target_center_y)
        self._smoothed_zoom = self._ema(self._smoothed_zoom, target_zoom)

        return self._build_crop_window(
            self._smoothed_center_x,
            self._smoothed_center_y,
            self._smoothed_zoom,
            detection.frame_timestamp_sec if detection else 0.0,
        )

    def _ema(self, previous: float | None, new_value: float) -> float:
        if previous is None:
            return new_value
        alpha = self._smoothing_alpha
        return previous * (1 - alpha) + new_value * alpha

    def _compute_zoom_factor(self, bbox: BoundingBox) -> float:
        frame_area = self._source_width * self._source_height
        subject_area_ratio = bbox.area / frame_area if frame_area else 0.0

        if subject_area_ratio >= self._min_subject_area_ratio or subject_area_ratio <= 0:
            return 1.0

        # Чем меньше объект относительно порога — тем сильнее зум, но не выше потолка.
        raw_zoom = (self._min_subject_area_ratio / subject_area_ratio) ** 0.5
        return min(raw_zoom, self._max_zoom_factor)

    def _build_crop_window(
        self, center_x: float, center_y: float, zoom_factor: float, timestamp_sec: float
    ) -> CropWindow:
        # Базовое окно кропа — самое большое окно с целевым аспект-рейшо,
        # которое помещается в оригинальный кадр, затем делится на zoom_factor.
        if self._source_width / self._source_height > self._target_aspect_ratio:
            base_height = self._source_height
            base_width = base_height * self._target_aspect_ratio
        else:
            base_width = self._source_width
            base_height = base_width / self._target_aspect_ratio

        crop_width = base_width / zoom_factor
        crop_height = base_height / zoom_factor

        x1 = center_x - crop_width / 2
        y1 = center_y - crop_height / 2
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # Клэмп в границы оригинального кадра со сдвигом окна (не сжатием) —
        # иначе объект у самого края кадра давал бы искажённый по размеру кроп.
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > self._source_width:
            x1 -= x2 - self._source_width
            x2 = self._source_width
        if y2 > self._source_height:
            y1 -= y2 - self._source_height
            y2 = self._source_height

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)

        return CropWindow(x1=x1, y1=y1, x2=x2, y2=y2, zoom_factor=zoom_factor, timestamp_sec=timestamp_sec)
