"""Из чего сложилась оценка Viral Score: вклад каждого сигнала (для окна и для момента)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalPart:
    key: str
    label: str
    value: float          # 0..1: насколько выражен сигнал
    weight: float         # вес из настроек
    points: float         # вклад в итоговые 0..100
    detail: str = ""      # «лай, смех», «2 прыжка»
    bonus: bool = False   # не входит в нормировку (motion_events)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "value": round(self.value, 3), "weight": self.weight,
                "points": round(self.points, 2), "detail": self.detail, "bonus": self.bonus}

    @classmethod
    def from_dict(cls, data: dict) -> "SignalPart":
        return cls(str(data["key"]), str(data.get("label") or data["key"]), float(data.get("value", 0.0)), float(data.get("weight", 0.0)),
                   float(data.get("points", 0.0)), str(data.get("detail") or ""), bool(data.get("bonus", False)))


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    score: int
    parts: tuple[SignalPart, ...]

    def part(self, key: str) -> SignalPart | None:
        return next((p for p in self.parts if p.key == key), None)
