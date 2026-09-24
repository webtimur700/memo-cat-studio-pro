"""Выбор шрифта с реальной проверкой покрытия символов.

РЕАЛЬНАЯ ПРОБЛЕМА, пойманная тестом на этом же шаге: многие "модные" шрифты
(Poppins-Bold и подобные Google Fonts) не содержат кириллицу — при рендере
русского текста Pillow рисует ".notdef" (пустой квадрат/tofu), это НЕ бросает
исключение и НЕ даёт пустой bbox (сам tofu-глиф имеет геометрию), поэтому
"глазами" через getmask().getbbox() эту проблему не поймать — нужна честная
проверка таблицы cmap через fontTools.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont

# Кодовые диапазоны эмодзи (основные блоки) — эмодзи всегда рендерятся
# отдельным цветным шрифтом (NotoColorEmoji), поэтому их не учитываем при
# проверке покрытия текстового шрифта.
_EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2190, 0x21FF),
    (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F),  # variation selectors
]


def is_emoji(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in _EMOJI_RANGES)


@lru_cache(maxsize=16)
def _load_cmap(font_path: str) -> frozenset[int]:
    font = TTFont(font_path, lazy=True)
    cmap = font.getBestCmap()
    return frozenset(cmap.keys())


def font_supports_text(font_path: str, text: str) -> bool:
    """True, если шрифт реально содержит глиф для КАЖДОГО не-эмодзи символа
    (пробелы и переводы строк игнорируются — они не требуют глифа)."""
    try:
        covered = _load_cmap(font_path)
    except Exception:
        return False

    for char in text:
        if char.isspace() or is_emoji(char):
            continue
        if ord(char) not in covered:
            return False
    return True


def resolve_font_path(text: str, candidates: list[str], fallback: str) -> str:
    """Возвращает первый шрифт из candidates, реально покрывающий весь текст;
    если ни один не подходит — гарантированный fallback (должен покрывать
    минимум кириллицу+латиницу, см. DEFAULT_FALLBACK_FONT).
    """
    for candidate in candidates:
        if font_supports_text(candidate, text):
            return candidate
    return fallback


def _fc_match(pattern: str) -> str | None:
    """Путь к шрифту по fontconfig-паттерну (переносимо между дистрибутивами:
    в Fedora нет /usr/share/fonts/truetype/dejavu/, как в Debian)."""
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = result.stdout.strip()
    return path if path and Path(path).is_file() else None


def _first_existing(paths: list[str], fc_pattern: str) -> str:
    for path in paths:
        if Path(path).is_file():
            return path
    return _fc_match(fc_pattern) or paths[0]


# Bold-шрифт с кириллицей+латиницей: известные пути Debian/Fedora, затем fontconfig.
DEFAULT_FALLBACK_FONT = _first_existing(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ],
    "sans-serif:bold",
)
DEFAULT_EMOJI_FONT = _first_existing(
    [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/google-noto-color-emoji-fonts/Noto-COLRv1.ttf",
        "/usr/share/fonts/google-noto-emoji-color-fonts/NotoColorEmoji.ttf",
    ],
    "Noto Color Emoji",
)
