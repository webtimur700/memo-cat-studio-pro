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


def find_font_file(family: str, bold: bool = True) -> str | None:
    """Путь к файлу шрифта именно этого семейства (fontconfig), либо None.
    fc-match всегда что-то возвращает (подменяет отсутствующее семейство другим),
    поэтому проверяем, что семейство в ответе совпало с запрошенным.
    """
    pattern = f"{family}:bold" if bold else family
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}|%{file}", pattern],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    found_family, _, path = result.stdout.partition("|")
    if family.lower() not in found_family.lower() or not Path(path).is_file():
        return None
    return path


def preferred_title_fonts() -> list[str]:
    """Модные заголовочные шрифты, которые реально есть на этой машине (пути
    Debian-google-fonts и любые системные через fontconfig)."""
    candidates = [
        "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
        "/usr/share/fonts/truetype/google-fonts/Montserrat-Bold.ttf",
    ]
    for family in ("Poppins", "Montserrat"):
        path = find_font_file(family)
        if path:
            candidates.append(path)
    return [c for c in dict.fromkeys(candidates) if Path(c).is_file()]


_EMOJI_PROBE_SIZES = (109, 96, 128, 72, 64, 61, 160, 136, 48)


def _strike_sizes(font_path: str) -> list[int]:
    """Размеры bitmap-strike'ов (CBDT/sbix) — единственные, что принимает такой шрифт."""
    try:
        font = TTFont(font_path, lazy=True)
        if "CBLC" in font:
            return [int(st.bitmapSizeTable.ppemX) for st in font["CBLC"].strikes]
        if "sbix" in font:
            return [int(k) for k in font["sbix"].strikes.keys()]
    except Exception:
        pass
    return []


def _renders_emoji(font_path: str, size: int) -> bool:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(font_path, size)
        probe = Image.new("RGBA", (size * 3, size * 3), (0, 0, 0, 0))
        ImageDraw.Draw(probe).text((0, 0), "\U0001F525", font=font, embedded_color=True)
        return probe.getbbox() is not None
    except Exception:
        return False


@lru_cache(maxsize=1)
def find_emoji_font() -> tuple[str, int] | None:
    """(путь, размер) первого эмодзи-шрифта, который Pillow РЕАЛЬНО рисует.

    На Fedora системный Noto Color Emoji — COLRv1: Pillow открывает его без
    ошибок, но рисует пустоту. Поэтому кандидата принимаем только после
    пробного рендера 🔥 (bbox не пуст), иначе идём к следующему — например,
    к Twemoji (CBDT, один размер strike'а — берём его из таблицы CBLC).
    """
    candidates = [DEFAULT_EMOJI_FONT]
    for name in ("Noto Color Emoji", "Twemoji", "Apple Color Emoji"):
        path = _fc_match(name)
        if path:
            candidates.append(path)
    try:
        listing = subprocess.run(
            ["fc-list", ":lang=und-zsye", "file"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        candidates += [line.strip().rstrip(":") for line in listing.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass

    for path in dict.fromkeys(candidates):
        if not Path(path).is_file():
            continue
        sizes = _strike_sizes(path) or list(_EMOJI_PROBE_SIZES)
        for size in sizes:
            if _renders_emoji(path, size):
                return path, size
    return None
