"""Центральный реестр расширений. Ядро (effects/, scoring/, animation/)
читает из этого реестра ТОЛЬКО через явные lookup-методы (get_effect и т.д.)
— плагины не патчат внутренности ядра напрямую, что делает их безопасными
для смены/удаления без риска сломать что-то за пределами их собственного
пространства имён.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PluginRegistry:
    _effects: dict[str, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)
    _templates: dict[str, dict] = field(default_factory=dict)
    _animation_styles: dict[str, Callable] = field(default_factory=dict)
    _scoring_strategies: dict[str, Callable[[dict], float]] = field(default_factory=dict)

    def register_effect(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        if name in self._effects:
            raise ValueError(f"Эффект '{name}' уже зарегистрирован — имена плагинов должны быть уникальны")
        self._effects[name] = fn

    def get_effect(self, name: str) -> Callable[[np.ndarray], np.ndarray] | None:
        return self._effects.get(name)

    def list_effects(self) -> list[str]:
        return sorted(self._effects.keys())

    def register_template(self, name: str, template: dict) -> None:
        if name in self._templates:
            raise ValueError(f"Шаблон '{name}' уже зарегистрирован")
        self._templates[name] = template

    def get_template(self, name: str) -> dict | None:
        return self._templates.get(name)

    def register_animation_style(self, name: str, fn: Callable) -> None:
        if name in self._animation_styles:
            raise ValueError(f"Анимационный стиль '{name}' уже зарегистрирован")
        self._animation_styles[name] = fn

    def get_animation_style(self, name: str) -> Callable | None:
        return self._animation_styles.get(name)

    def register_scoring_strategy(self, name: str, fn: Callable[[dict], float]) -> None:
        if name in self._scoring_strategies:
            raise ValueError(f"Стратегия скоринга '{name}' уже зарегистрирована")
        self._scoring_strategies[name] = fn

    def get_scoring_strategy(self, name: str) -> Callable[[dict], float] | None:
        return self._scoring_strategies.get(name)
