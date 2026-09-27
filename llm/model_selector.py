"""Автовыбор LLM-модели: лучшая по замерам из тех, что помещаются в память.

Правила (в порядке приоритета):
  1. LM_STUDIO_MODEL_OVERRIDE из .env — всегда, без проверок.
  2. Embedding-модели исключаются: это не чат-модели (в /v1/models при JIT-загрузке
     они лежат в общем списке, поэтому "первая из списка" давала случайный выбор).
  3. Среди чат-моделей берётся лучшая по рейтингу (DEFAULT_PROFILES, составлен
     по замерам docs/llm_benchmark.md), которой хватает свободной памяти
     (MemAvailable + то, что вернёт выгрузка уже загруженных LLM) с запасом под
     остальной пайплайн (Whisper, YOLO, ffmpeg, Qt).
  4. Не влезла — запасной может быть только модель, которая МЕНЬШЕ (помещается) и НЕ МЕДЛЕННЕЕ
     лучшей из не влезших: по измеренным секундам на клип, иначе по числу активных параметров
     (26B-A4B -> 4B). Следующая по рейтингу, но медленнее, запасной не считается: так при нехватке
     0.2 ГиБ под Gemma (~30 с/клип) селектор брал плотную Qwen3.8-27B (140–190 с/клип, зависания).
     Нет такой — модели нет, клипы получают заголовки по умолчанию, интерфейс показывает причину.
  5. Память модели — измеренный пик (профиль), а не доля размера GGUF: у MoE пик заметно меньше
     файла, у плотной — больше (Qwen3.8: файл 16.5 ГиБ, пик с кадром до 16.8 ГиБ), поэтому по файлу
     плотная Qwen3.8 выглядела «меньше» Gemma (16.8 ГиБ). Для неизвестных — оценка по размеру файла.
  6. Модели, недавно не ответившие за таймаут (llm/model_health.py), идут после остальных.
  7. Модели вне рейтинга идут после известных: чем крупнее (из влезающих), тем лучше.
Решение и его причина возвращаются строкой — вызывающий пишет её в лог.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llm.lm_studio_api import ModelInfo

MIB = 1024**2

# Запас под остальной пайплайн, пока LLM загружена: Whisper (~1.5 ГиБ), YOLO+OpenCV,
# буферы ffmpeg/libx264 при кодировании 1080x1920, Qt-интерфейс, система.
DEFAULT_PIPELINE_RESERVE_MIB = 6 * 1024

# Оценка для модели без замера = доля размера GGUF + KV-кэш. Доля 1.0, а не меньше: у плотной модели пик
# с кадром достигает размера файла (Qwen3.8: 16.8 при файле 16.5 ГиБ), у MoE он меньше — завышение для
# неизвестной MoE безопаснее, чем занижение для неизвестной плотной.
FOOTPRINT_RATIO = 1.0
KV_CACHE_MIB = 1024
# Сколько вернёт выгрузка загруженной модели без замера: нижняя граница замеров (MemAvailable падает на
# 45–66% размера файла) — лучше недосчитать освобождаемое, чем пообещать память, которой нет.
UNLOAD_RETURN_RATIO = 0.45


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Что известно о модели из замеров. key_part — подстрока id модели (без регистра)."""

    key_part: str
    rank: int                       # 1 — лучшая; меньше число — выше приоритет
    reasoning_effort: str | None    # что передавать в reasoning_effort ("none" — без рассуждений)
    note: str = ""
    vision: bool = True             # можно ли отправлять кадр (у некоторых моделей с кадром ломается формат ответа)
    peak_mib: float | None = None       # измеренный пик падения MemAvailable: загрузка (контекст 8192) + запрос с кадром
    loaded_mib: float | None = None     # измеренное падение MemAvailable после загрузки — столько вернёт выгрузка
    sec_per_clip: float | None = None   # измеренное время текстов клипа (один запрос с кадром) в пайплайне


# Составлено по замерам docs/llm_benchmark.md (свой видео-материал, рассуждения выключены).
# Память (docs/llm_selection.md): peak_mib — наибольший измеренный пик падения MemAvailable с кадром (замер
# 2026-09-24 с контекстом 16384 — выше, чем при рабочих 8192, то есть с запасом); loaded_mib — падение после
# загрузки с контекстом 8192 (2026-09-27, scripts/measure_llm_memory.py): столько вернёт выгрузка.
# sec_per_clip — тексты клипа одним запросом с кадром в пайплайне (docs/llm_combined.md).
# Модели, которых здесь нет, идут после этих (не измерены) — по размеру.
GIB_MIB = 1024
DEFAULT_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        "gemma-4-26b-a4b", 1, "none",
        "быстрая (25 с/клип, с кадром 38 с), ровный формат в обоих режимах, самый низкий пик памяти, "
        "с кадром описывает реально видимое",
        peak_mib=11.8 * GIB_MIB, loaded_mib=7.4 * GIB_MIB, sec_per_clip=31,
    ),
    ModelProfile(
        "qwen3.6-35b-a3b", 2, "none",
        "быстрая (28–37 с/клип) и яркие заголовки, но самая тяжёлая (20.6 ГиБ, загрузка ~105 с); с кадром "
        "ломает формат (хештегов 14–21 вместо 30, короткое описание) — кадр не отправляется",
        vision=False,
        peak_mib=13.2 * GIB_MIB, loaded_mib=9.6 * GIB_MIB, sec_per_clip=33,
    ),
    ModelProfile(
        "qwen3.8-27b", 3, "none",
        "плотная 27B: 124–194 с/клип с кадром, пик памяти до 16.8 ГиБ без выигрыша в качестве, зависала на кадрах",
        peak_mib=16.8 * GIB_MIB, loaded_mib=9.8 * GIB_MIB, sec_per_clip=170,
    ),
)


@dataclass(frozen=True, slots=True)
class Selection:
    model_key: str
    reasoning_effort: str | None
    supports_vision: bool
    reason: str
    already_loaded: bool = False
    from_override: bool = False


class NoSuitableModelError(RuntimeError):
    """Подходящей модели нет. Для «не хватает памяти» заполнены цифры самой лёгкой модели — по ним
    интерфейс показывает, сколько нужно и сколько свободно; для «нет чат-моделей» цифр нет."""

    def __init__(
        self, message: str, need_mib: float | None = None, available_mib: float | None = None,
        reserve_mib: float | None = None, lightest_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.need_mib = need_mib
        self.available_mib = available_mib
        self.reserve_mib = reserve_mib
        self.lightest_key = lightest_key


def read_mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo_path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable не найден в /proc/meminfo")


def _profile_for(key: str, profiles: tuple[ModelProfile, ...]) -> ModelProfile | None:
    lowered = key.lower()
    for profile in profiles:
        if profile.key_part.lower() in lowered:
            return profile
    return None


def estimate_footprint_mib(model: ModelInfo, profiles: tuple[ModelProfile, ...] | None = None) -> float:
    """Сколько памяти модель займёт на пике: измеренное (профиль) или оценка по размеру файла."""
    profile = _profile_for(model.key, DEFAULT_PROFILES if profiles is None else profiles)
    if profile is not None and profile.peak_mib is not None:
        return profile.peak_mib
    size_mib = (model.size_bytes or 0) / MIB
    return size_mib * FOOTPRINT_RATIO + KV_CACHE_MIB


def returned_by_unload_mib(model: ModelInfo, profiles: tuple[ModelProfile, ...] | None = None) -> float:
    """Сколько MemAvailable вернёт выгрузка этой (загруженной) модели."""
    profile = _profile_for(model.key, DEFAULT_PROFILES if profiles is None else profiles)
    if profile is not None and profile.loaded_mib is not None:
        return profile.loaded_mib
    return (model.size_bytes or 0) / MIB * UNLOAD_RETURN_RATIO


_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*B", re.IGNORECASE)


def active_params_b(model: ModelInfo) -> float | None:
    """Активные параметры, млрд: "26B-A4B" -> 4 (MoE), "27B" -> 27 (плотная). Скорость генерации растёт
    обратно им, поэтому это мерило скорости для моделей без замера. Нет params_string — None."""
    if not model.params:
        return None
    found = [float(x) for x in _PARAMS.findall(model.params)]
    if not found:
        return None
    moe = re.search(r"A(\d+(?:\.\d+)?)\s*B", model.params, re.IGNORECASE)
    return float(moe.group(1)) if moe else found[0]


def _speed_hint(model: ModelInfo, profile: ModelProfile | None) -> str:
    if profile is not None and profile.sec_per_clip is not None:
        return f"~{profile.sec_per_clip:.0f} с/клип"
    active = active_params_b(model)
    return f"{active:g}B активных" if active is not None else "скорость неизвестна"


def not_slower(candidate: ModelInfo, than: ModelInfo, profiles: tuple[ModelProfile, ...]) -> bool:
    """Запасная модель не должна быть медленнее той, которую заменяет. Сравнение по замерам, иначе по
    активным параметрам; если сравнить нечем — допускается (о скорости ничего не известно)."""
    cand_p, than_p = _profile_for(candidate.key, profiles), _profile_for(than.key, profiles)
    if cand_p and than_p and cand_p.sec_per_clip is not None and than_p.sec_per_clip is not None:
        return cand_p.sec_per_clip <= than_p.sec_per_clip
    cand_a, than_a = active_params_b(candidate), active_params_b(than)
    if cand_a is not None and than_a is not None:
        return cand_a <= than_a
    return True


def default_reasoning_effort(model: ModelInfo) -> str | None:
    """Для модели вне рейтинга: рассуждения выключаем, если модель это позволяет
    (замеры: рассуждения увеличивают время клипа с секунд до минут без выигрыша
    в качестве коротких заголовков)."""
    return "none" if model.supports_reasoning_control else None


def available_after_unload_mib(
    models: list[ModelInfo], mem_available_mib: float, profiles: tuple[ModelProfile, ...] | None = None
) -> float:
    """MemAvailable + память, которую вернёт выгрузка уже загруженных LLM (по замеру, иначе нижняя оценка).
    Загруженная модель «бесплатной» не считается: её нужная память в select_model полная, а занятое ею
    возвращается сюда — только так уже загруженная тяжёлая модель не выигрывает у лёгкой на пустом месте."""
    return mem_available_mib + sum(returned_by_unload_mib(m, profiles) for m in models if m.loaded_instances)


def select_model(
    models: list[ModelInfo],
    available_mib: float,
    override: str | None = None,
    reserve_mib: float = DEFAULT_PIPELINE_RESERVE_MIB,
    profiles: tuple[ModelProfile, ...] | None = None,
    penalized: frozenset[str] | set[str] = frozenset(),
) -> Selection:
    """penalized — модели, недавно не ответившие за таймаут: идут после остальных (не исключаются)."""
    profiles = DEFAULT_PROFILES if profiles is None else profiles
    by_key = {m.key: m for m in models}

    if override:
        info = by_key.get(override)
        profile = _profile_for(override, profiles)
        effort = profile.reasoning_effort if profile else (default_reasoning_effort(info) if info else None)
        return Selection(
            model_key=override,
            reasoning_effort=effort,
            supports_vision=bool(info and info.vision and (profile.vision if profile else True)),
            reason=f"задана вручную (LM_STUDIO_MODEL_OVERRIDE={override})",
            already_loaded=bool(info and info.loaded_instances),
            from_override=True,
        )

    chat_models = [m for m in models if m.is_chat_model and "embed" not in m.key.lower()]
    excluded = [m.key for m in models if m not in chat_models]
    if not chat_models:
        raise NoSuitableModelError(
            "В LM Studio нет чат-моделей (только embedding). Скачайте LLM во вкладке Discover."
        )

    def need(model: ModelInfo) -> float:
        return estimate_footprint_mib(model, profiles)

    def sort_key(model: ModelInfo) -> tuple[bool, int, float, float]:
        profile = _profile_for(model.key, profiles)
        demoted = model.key in penalized
        if profile:
            return (demoted, 0, profile.rank, 0.0)
        return (demoted, 1, 0.0, -(model.size_bytes or 0))   # неизвестные — после известных, крупные раньше

    ranked = sorted(chat_models, key=sort_key)
    budget_mib = available_mib - reserve_mib
    skipped: list[str] = []
    too_slow: list[str] = []
    replaced: ModelInfo | None = None     # лучшая по рейтингу из не влезших: запасная не должна быть медленнее её

    for place, model in enumerate(ranked, start=1):
        need_mib = need(model)
        profile = _profile_for(model.key, profiles)
        if need_mib > budget_mib:
            skipped.append(f"{model.key} (нужно ~{need_mib / 1024:.1f} ГиБ > {budget_mib / 1024:.1f})")
            replaced = replaced or model
            continue
        if replaced is not None and not not_slower(model, replaced, profiles):
            too_slow.append(
                f"{model.key} (помещается, но медленнее {replaced.key}: "
                f"{_speed_hint(model, profile)} против {_speed_hint(replaced, _profile_for(replaced.key, profiles))})"
            )
            continue
        effort = profile.reasoning_effort if profile else default_reasoning_effort(model)
        rating = (
            ("запасная вне рейтинга (не измерена), лучшая из помещающихся (меньше и не медленнее)" if replaced else "вне рейтинга (не измерена)")
            if profile is None
            else "лучшая в рейтинге" if replaced is None else "лучшая из помещающихся (меньше и не медленнее)"
        )
        reason = (
            f"{'уже загружена; ' if model.loaded_instances else ''}"
            f"{rating} (место {place} из {len(ranked)}), "
            f"нужно ~{need_mib / 1024:.1f} ГиБ, доступно {available_mib / 1024:.1f} ГиБ "
            f"при резерве {reserve_mib / 1024:.1f} ГиБ под пайплайн"
        )
        if skipped:
            reason += "; не поместились: " + ", ".join(skipped)
        if too_slow:
            reason += "; не взяты: " + ", ".join(too_slow)
        demoted = [m.key for m in ranked if m.key in penalized]
        if demoted:
            reason += "; понижены за недавний таймаут: " + ", ".join(demoted)
        if excluded:
            reason += "; исключены не-чат модели: " + ", ".join(excluded)
        return Selection(
            model_key=model.key,
            reasoning_effort=effort,
            supports_vision=model.vision and (profile.vision if profile else True),
            reason=reason,
            already_loaded=bool(model.loaded_instances),
        )

    # Ничего не подошло. Цифры для интерфейса — по самой лёгкой из допустимых (медленные запасные не в счёт):
    # именно ей нужно освободить память.
    allowed = [m for m in chat_models if replaced is None or m is replaced or not_slower(m, replaced, profiles)]
    lightest = min(allowed, key=need)
    message = (
        f"Ни одна подходящая LLM не помещается в память: доступно {available_mib / 1024:.1f} ГиБ, "
        f"резерв под пайплайн {reserve_mib / 1024:.1f} ГиБ. " + "; ".join(skipped)
    )
    if too_slow:
        message += "; не взяты как запасные: " + "; ".join(too_slow)
    raise NoSuitableModelError(
        message, need_mib=need(lightest), available_mib=available_mib, reserve_mib=reserve_mib, lightest_key=lightest.key,
    )
