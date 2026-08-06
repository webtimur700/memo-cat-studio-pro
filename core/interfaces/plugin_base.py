"""Контракт плагина. Любой плагин в plugins/ — это Python-пакет с файлом
`plugin.py`, экспортирующим объект `PLUGIN`, реализующий этот протокол.
plugins/loader.py находит такие пакеты и вызывает register() на каждом —
это единственная точка, где плагин взаимодействует с ядром приложения.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from plugins.sdk.registry import PluginRegistry


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str = "unknown"


class PluginBase(Protocol):
    def metadata(self) -> PluginMetadata:
        ...

    def register(self, registry: PluginRegistry) -> None:
        """Регистрирует то, что предоставляет плагин (эффект/шаблон/анимацию/
        стратегию скоринга) в общем реестре — единственный способ плагину
        повлиять на пайплайн, без прямого импорта или правки ядра.
        """
        ...
