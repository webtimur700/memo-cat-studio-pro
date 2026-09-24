"""Безопасная зона кадра Shorts: прямоугольник, в котором ничего не перекрывается
интерфейсом YouTube (название/канал снизу, кнопки справа, поиск сверху)."""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.detection import BoundingBox
from core.entities.settings import SafeZoneSettings


@dataclass(frozen=True, slots=True)
class SafeZone:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_box(self) -> BoundingBox:
        return BoundingBox(float(self.x1), float(self.y1), float(self.x2), float(self.y2))

    def with_bottom(self, y2: int) -> "SafeZone":
        """Та же зона, но с другой нижней границей (например, над субтитрами)."""
        return SafeZone(self.x1, self.y1, self.x2, max(self.y1 + 1, min(self.y2, y2)))

    @classmethod
    def from_settings(cls, frame_size: tuple[int, int], margins: SafeZoneSettings) -> "SafeZone":
        width, height = frame_size
        # масштабируем поля, заданные для 1080x1920, если экспорт в другом разрешении
        scale = width / 1080
        x1 = round(margins.left_px * scale)
        x2 = width - round(margins.right_px * scale)
        y1 = round(margins.top_px * scale)
        y2 = height - round(margins.bottom_px * scale)
        if x2 - x1 < width * 0.3 or y2 - y1 < height * 0.3:   # заведомо ошибочные поля — не душим кадр
            return cls(0, 0, width, height)
        return cls(x1, y1, x2, y2)

    def subtitle_margins(self, frame_size: tuple[int, int]) -> tuple[int, int, int]:
        """(margin_l, margin_r, margin_v) для ASS: расстояния от краёв кадра до текста."""
        width, height = frame_size
        return self.x1, width - self.x2, height - self.y2
