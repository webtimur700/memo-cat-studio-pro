"""Collision detection для рекламной плашки (Функция 12: "если AI обнаружил,
что плашка закрывает животное — автоматически изменить положение").

Геометрия: bbox плашки × "головная зона" животного (vision/mediapipe_face.
AnimalHeadRegionEstimator) — если IoU/перекрытие выше порога, предлагается
альтернативная позиция плашки из заранее заданного списка кандидатов
(верх/низ/левый/правый край кадра), выбирается первая БЕЗ пересечения.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.entities.detection import BoundingBox

# Небольшой порог перекрытия достаточен — плашка не должна перекрывать
# животное даже частично, это не "выбираем наименьшее зло", а бинарный тест.
MIN_OVERLAP_AREA_RATIO_FOR_COLLISION = 0.02


class BannerPosition(str, Enum):
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"


# Порядок предпочтения при поиске свободной позиции — сначала пробуем
# исходно настроенную пользователем (Функция 19 "стиль плашки"), затем
# ближайшие альтернативы.
_FALLBACK_ORDER: dict[BannerPosition, list[BannerPosition]] = {
    BannerPosition.BOTTOM_CENTER: [
        BannerPosition.BOTTOM_LEFT,
        BannerPosition.BOTTOM_RIGHT,
        BannerPosition.TOP_CENTER,
    ],
    BannerPosition.TOP_CENTER: [
        BannerPosition.TOP_LEFT,
        BannerPosition.TOP_RIGHT,
        BannerPosition.BOTTOM_CENTER,
    ],
    BannerPosition.BOTTOM_LEFT: [BannerPosition.BOTTOM_RIGHT, BannerPosition.TOP_LEFT],
    BannerPosition.BOTTOM_RIGHT: [BannerPosition.BOTTOM_LEFT, BannerPosition.TOP_RIGHT],
    BannerPosition.TOP_LEFT: [BannerPosition.TOP_RIGHT, BannerPosition.BOTTOM_LEFT],
    BannerPosition.TOP_RIGHT: [BannerPosition.TOP_LEFT, BannerPosition.BOTTOM_RIGHT],
}


def _overlap_area_ratio(a: BoundingBox, b: BoundingBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    return inter_area / a.area if a.area > 0 else 0.0


def compute_banner_rect(
    position: BannerPosition,
    frame_width: int,
    frame_height: int,
    banner_width: int,
    banner_height: int,
    margin: int = 40,
) -> BoundingBox:
    if position in (BannerPosition.BOTTOM_LEFT, BannerPosition.TOP_LEFT):
        x1 = margin
    elif position in (BannerPosition.BOTTOM_RIGHT, BannerPosition.TOP_RIGHT):
        x1 = frame_width - banner_width - margin
    else:  # *_CENTER
        x1 = (frame_width - banner_width) // 2

    if position in (BannerPosition.TOP_CENTER, BannerPosition.TOP_LEFT, BannerPosition.TOP_RIGHT):
        y1 = margin
    else:
        y1 = frame_height - banner_height - margin

    return BoundingBox(x1=x1, y1=y1, x2=x1 + banner_width, y2=y1 + banner_height)


@dataclass(frozen=True, slots=True)
class CollisionResolution:
    final_position: BannerPosition
    final_rect: BoundingBox
    was_repositioned: bool


def resolve_banner_position(
    preferred_position: BannerPosition,
    animal_head_region: BoundingBox | None,
    frame_width: int,
    frame_height: int,
    banner_width: int,
    banner_height: int,
) -> CollisionResolution:
    candidates = [preferred_position] + _FALLBACK_ORDER.get(preferred_position, [])

    for index, position in enumerate(candidates):
        rect = compute_banner_rect(position, frame_width, frame_height, banner_width, banner_height)

        if animal_head_region is None:
            return CollisionResolution(position, rect, was_repositioned=index > 0)

        overlap = _overlap_area_ratio(rect, animal_head_region)
        if overlap < MIN_OVERLAP_AREA_RATIO_FOR_COLLISION:
            return CollisionResolution(position, rect, was_repositioned=index > 0)

    # Ни одна позиция не свободна (редкий случай — крупное животное почти
    # во весь кадр) — возвращаем исходную предпочтительную, ничего лучше нет.
    fallback_rect = compute_banner_rect(
        preferred_position, frame_width, frame_height, banner_width, banner_height
    )
    return CollisionResolution(preferred_position, fallback_rect, was_repositioned=False)
