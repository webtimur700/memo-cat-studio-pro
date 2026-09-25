"""Сохранение настроек в SQLite: в базе лежат только поля, которые пользователь изменил относительно
config/default_settings.yaml (по строке на поле). Поэтому новые значения по умолчанию из yaml доходят до всех
полей, которых пользователь не трогал, а сброс поля к значению по умолчанию удаляет его строку.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from typing import Any

from loguru import logger

from core.entities.settings import UserSettings
from database.db import Database


def _to_json_value(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _from_json_value(default: Any, value: Any) -> Any:
    """Приводит значение из JSON к типу поля по значению по умолчанию; несовместимое — None (пропускается)."""
    if isinstance(default, tuple):
        return tuple(value) if isinstance(value, list) else None
    if isinstance(default, bool):
        return value if isinstance(value, bool) else None
    if isinstance(default, int):
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(default, float):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if isinstance(default, str):
        return value if isinstance(value, str) else None
    return None


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def load(self, base: UserSettings) -> UserSettings:
        """base (yaml) + сохранённые пользователем поля."""
        result = base
        for row in self._db.query("SELECT section, key, value FROM settings"):
            section_name, key = row["section"], row["key"]
            section = getattr(result, section_name, None)
            if section is None or not is_dataclass(section) or key not in {f.name for f in fields(section)}:
                logger.warning("Настройки: неизвестное поле {}.{} в базе — пропущено", section_name, key)
                continue
            try:
                value = _from_json_value(getattr(section, key), json.loads(row["value"]))
            except json.JSONDecodeError:
                value = None
            if value is None:
                logger.warning("Настройки: значение {}.{} в базе не подходит по типу — пропущено", section_name, key)
                continue
            result = replace(result, **{section_name: replace(section, **{key: value})})
        return result

    def save(self, settings: UserSettings, base: UserSettings) -> int:
        """Записывает поля, отличающиеся от base; равные base — удаляет. Возвращает число сохранённых полей."""
        changed: list[tuple[str, str, str]] = []
        for section_field in fields(settings):
            section, base_section = getattr(settings, section_field.name), getattr(base, section_field.name)
            if not is_dataclass(section):
                continue
            for f in fields(section):
                value, default = getattr(section, f.name), getattr(base_section, f.name)
                if value != default:
                    changed.append((section_field.name, f.name, json.dumps(_to_json_value(value), ensure_ascii=False)))
        statements = [("DELETE FROM settings", ())] + [
            ("INSERT INTO settings(section, key, value) VALUES (?, ?, ?)", row) for row in changed
        ]
        self._db.transaction(statements)   # одной транзакцией: не остаётся полузаписанных настроек
        return len(changed)
