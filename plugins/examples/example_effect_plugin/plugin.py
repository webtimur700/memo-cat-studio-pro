"""Пример плагина: добавляет эффект "vintage_sepia" — сепия-тонирование
кадра. Настоящая, работающая реализация (матрица преобразования цвета), а
не заглушка — задумана как шаблон для пользовательских плагинов эффектов.

Чтобы написать свой плагин: скопировать эту директорию, изменить
metadata()/register(), заменить apply_vintage_sepia на свою функцию с той же
сигнатурой (np.ndarray BGR -> np.ndarray BGR).
"""

from __future__ import annotations

import numpy as np

from core.interfaces.plugin_base import PluginMetadata
from plugins.sdk.registry import PluginRegistry

# Стандартная sepia-матрица (веса каналов BGR -> BGR с эффектом сепии).
_SEPIA_MATRIX = np.array([
    [0.131, 0.534, 0.272],   # выходной B
    [0.168, 0.686, 0.349],   # выходной G
    [0.189, 0.769, 0.393],   # выходной R
])


def apply_vintage_sepia(frame_bgr: np.ndarray) -> np.ndarray:
    sepia = frame_bgr.astype(np.float32) @ _SEPIA_MATRIX.T
    return np.clip(sepia, 0, 255).astype(np.uint8)


class ExampleEffectPlugin:
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="vintage_sepia",
            version="1.0.0",
            description="Сепия-тонирование кадра — пример эффект-плагина",
            author="Memo Cat AI Studio Pro (пример)",
        )

    def register(self, registry: PluginRegistry) -> None:
        registry.register_effect("vintage_sepia", apply_vintage_sepia)


PLUGIN = ExampleEffectPlugin()
