"""Рендер субтитров в формат ASS (Advanced SubStation Alpha) с подсветкой
текущего слова (Функция 7: "Подсветка текущего слова").

Механизм подсветки — не отдельный кадр на каждое слово (что раздуло бы файл
и услож нило burn-in), а стандартный ASS-тег `\\k<centisecs>` (karaoke) в
сочетании с `\\1c&Hxxxxxx&` для цвета подсвеченного слова — это ровно тот
механизм, которым рендерят "подсветку слова" все современные Shorts-редакторы,
не изобретаем велосипед.

Экспорт в SRT (Функция 7) — отдельная, более простая функция без karaoke-тегов
(SRT формат их не поддерживает), сегмент показывается целиком.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.subtitle import SubtitleSegment

ASS_HEADER_TEMPLATE = """[Script Info]
Title: Memo Cat AI Studio Pro — Burned Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},{highlight_color},{outline_color},{back_color},{bold},0,0,0,100,100,0,0,1,{outline_width},{shadow},2,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@dataclass(frozen=True, slots=True)
class SubtitleStyle:
    """Один из пресетов из config/default_settings.yaml
    (modern_bold / minimal_clean / neon_pop / classic_yellow).
    Цвета в формате ASS — &HAABBGGRR& (порядок каналов ОБРАТНЫЙ обычному RGB,
    это особенность формата ASS, не опечатка).
    """

    name: str
    font_name: str = "Montserrat ExtraBold"
    font_size: int = 72
    primary_color: str = "&H00FFFFFF&"     # белый — обычный цвет текста
    highlight_color: str = "&H0000D7FF&"   # жёлто-оранжевый — цвет текущего слова
    outline_color: str = "&H00000000&"     # чёрная обводка
    back_color: str = "&H00000000&"
    bold: int = 1
    outline_width: float = 4.0
    shadow: float = 1.0
    margin_v: int = 220


STYLE_PRESETS: dict[str, SubtitleStyle] = {
    "modern_bold": SubtitleStyle(
        name="modern_bold",
        font_name="Montserrat ExtraBold",
        font_size=78,
        primary_color="&H00FFFFFF&",
        highlight_color="&H0000D7FF&",
        outline_width=5.0,
    ),
    "minimal_clean": SubtitleStyle(
        name="minimal_clean",
        font_name="Inter",
        font_size=60,
        primary_color="&H00FFFFFF&",
        highlight_color="&H00CCCCCC&",
        outline_width=2.0,
        bold=0,
    ),
    "neon_pop": SubtitleStyle(
        name="neon_pop",
        font_name="Poppins ExtraBold",
        font_size=80,
        primary_color="&H00FFFFFF&",
        highlight_color="&H00FF5CFF&",  # неоновый розовый
        outline_color="&H00FF5C7C&",
        outline_width=6.0,
    ),
    "classic_yellow": SubtitleStyle(
        name="classic_yellow",
        font_name="Arial Black",
        font_size=70,
        primary_color="&H0000FFFF&",   # жёлтый как основной цвет (классика YouTube)
        highlight_color="&H00FFFFFF&",
        outline_width=4.0,
    ),
}


def _seconds_to_ass_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int(round((secs - int(secs)) * 100))
    return f"{hours:d}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"


def _seconds_to_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int(round((secs - int(secs)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


def _build_karaoke_text(segment: SubtitleSegment, style: SubtitleStyle) -> str:
    """Каждое слово получает свой \\k-тег с длительностью в сантисекундах —
    плеер/burn-in подсвечивает слово ровно на это время, дальше подсветка
    переходит на следующее по тегу.
    """
    parts: list[str] = []
    for word in segment.words:
        duration_centisec = max(1, int(round((word.end_sec - word.start_sec) * 100)))
        # \\1c меняет цвет ПОСЛЕ karaoke-перехода — стандартный трюк для
        # "подсветка текущего слова, обычный цвет для остальных" в ASS.
        parts.append(f"{{\\k{duration_centisec}\\1c{style.highlight_color}}}{word.text} ")
    return "".join(parts).strip()


def render_ass(
    segments: list[SubtitleSegment],
    style_preset: str = "modern_bold",
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    margin_l: int = 40,
    margin_r: int = 40,
    margin_v: int | None = None,
) -> str:
    """margin_l/margin_r/margin_v — расстояния от краёв кадра до текста (для
    безопасной зоны Shorts, см. effects/safe_zone.py); margin_v=None — из пресета."""
    style = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["modern_bold"])

    header = ASS_HEADER_TEMPLATE.format(
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=style.font_name,
        font_size=style.font_size,
        primary_color=style.primary_color,
        highlight_color=style.primary_color,  # базовый цвет строки — обычный (не highlight)
        outline_color=style.outline_color,
        back_color=style.back_color,
        bold=style.bold,
        outline_width=style.outline_width,
        shadow=style.shadow,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=style.margin_v if margin_v is None else margin_v,
    )

    lines: list[str] = []
    for segment in segments:
        start_ts = _seconds_to_ass_timestamp(segment.start_sec)
        end_ts = _seconds_to_ass_timestamp(segment.end_sec)
        text = _build_karaoke_text(segment, style)
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


def estimate_subtitle_band_height(style_preset: str = "modern_bold", lines: int = 2) -> int:
    """Высота (px) полосы субтитров в `lines` строк — для резервирования места под неё."""
    style = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["modern_bold"])
    return int(style.font_size * 1.25 * lines + style.outline_width * 2)


def render_srt(segments: list[SubtitleSegment]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start_ts = _seconds_to_srt_timestamp(segment.start_sec)
        end_ts = _seconds_to_srt_timestamp(segment.end_sec)
        lines.append(f"{index}\n{start_ts} --> {end_ts}\n{segment.text}\n")
    return "\n".join(lines)
