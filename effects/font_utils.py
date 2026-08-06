"""Выбор шрифта с реальной проверкой покрытия символов.

РЕАЛЬНАЯ ПРОБЛЕМА, пойманная тестом на этом же шаге: многие "модные" шрифты
(Poppins-Bold и подобные Google Fonts) не содержат кириллицу — при рендере
русского текста Pillow рисует ".notdef" (пустой квадрат/tofu), это НЕ бросает
исключение и НЕ даёт пустой bbox (сам tofu-глиф имеет геометрию), поэтому
"глазами" через getmask().getbbox() эту проблему не поймать — нужна честная
проверка таблицы cmap через fontTools.
"""

from __future__ import annotations

from functools import lru_cache

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


# DejaVu Sans Bold — практически гарантированно предустановлен на любом Linux
# (в т.ч. внутри distrobox-образов Fedora) и покрывает кириллицу+латиницу+
# базовую пунктуацию — безопасный fallback последней инстанции.
DEFAULT_FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEFAULT_EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
