"""Загрузчик плагинов (Функция: "модульная система плагинов... без изменения
ядра приложения"). Сканирует директорию на подпакеты с файлом `plugin.py`,
импортирует их через importlib (не через статический import — иначе ядро
пришлось бы менять при каждом новом плагине) и вызывает `register()` каждого
найденного `PLUGIN` на переданном PluginRegistry.

Один сломанный плагин НЕ должен ронять загрузку остальных — ошибки
логируются и плагин пропускается, а не прерывает весь процесс.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from loguru import logger

from core.interfaces.plugin_base import PluginBase, PluginMetadata
from plugins.sdk.registry import PluginRegistry

PLUGIN_ENTRY_FILENAME = "plugin.py"


class PluginLoadError(Exception):
    pass


def _import_plugin_module(plugin_dir: Path):
    entry_path = plugin_dir / PLUGIN_ENTRY_FILENAME
    if not entry_path.exists():
        raise PluginLoadError(f"{plugin_dir} не содержит {PLUGIN_ENTRY_FILENAME}")

    module_name = f"memo_cat_plugin_{plugin_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Не удалось создать import spec для {entry_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_plugin_dirs(plugins_root: Path) -> list[Path]:
    if not plugins_root.exists():
        return []
    return sorted(
        p for p in plugins_root.iterdir()
        if p.is_dir() and (p / PLUGIN_ENTRY_FILENAME).exists()
    )


def load_plugins(plugins_root: Path, registry: PluginRegistry) -> list[PluginMetadata]:
    loaded: list[PluginMetadata] = []

    for plugin_dir in discover_plugin_dirs(plugins_root):
        try:
            module = _import_plugin_module(plugin_dir)
            plugin: PluginBase = getattr(module, "PLUGIN")
            metadata = plugin.metadata()
            plugin.register(registry)
            loaded.append(metadata)
            logger.info("Плагин загружен: {} v{} ({})", metadata.name, metadata.version, plugin_dir.name)
        except Exception as exc:
            # Осознанно широкий except: плагин — код от третьей стороны,
            # любая его ошибка (включая опечатки в самом плагине) не должна
            # останавливать загрузку остальных плагинов или приложения целиком.
            logger.warning("Не удалось загрузить плагин из {}: {}", plugin_dir, exc)

    return loaded
