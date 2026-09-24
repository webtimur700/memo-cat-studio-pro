"""Автовыбор LLM-модели: лучшая по замерам из тех, что помещаются в память.

Правила (в порядке приоритета):
  1. LM_STUDIO_MODEL_OVERRIDE из .env — всегда, без проверок.
  2. Embedding-модели исключаются: это не чат-модели (в /v1/models при JIT-загрузке
     они лежат в общем списке, поэтому "первая из списка" давала случайный выбор).
  3. Среди чат-моделей берётся лучшая по рейтингу (DEFAULT_PROFILES, составлен
     по замерам docs/llm_benchmark.md), которой хватает свободной памяти
     (MemAvailable) с запасом под остальной пайплайн (Whisper, YOLO, ffmpeg, Qt).
     Не влезла — берётся следующая, меньшая.
  4. Модели вне рейтинга идут после известных: чем крупнее (из влезающих), тем лучше.
Решение и его причина возвращаются строкой — вызывающий пишет её в лог.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm.lm_studio_api import ModelInfo

MIB = 1024**2

# Запас под остальной пайплайн, пока LLM загружена: Whisper (~1.5 ГиБ), YOLO+OpenCV,
# буферы ffmpeg/libx264 при кодировании 1080x1920, Qt-интерфейс, система.
DEFAULT_PIPELINE_RESERVE_MIB = 6 * 1024

# Оценка занятой памяти = доля размера GGUF (часть весов остаётся в page cache/GTT и
# вытесняется по необходимости, поэтому MemAvailable падает меньше размера файла:
# замеры 50–66%, пик при генерации с кадром до ~80%) + KV-кэш выбранного контекста.
FOOTPRINT_RATIO = 0.8
KV_CACHE_MIB = 1024


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Что известно о модели из замеров. key_part — подстрока id модели (без регистра)."""

    key_part: str
    rank: int                       # 1 — лучшая; меньше число — выше приоритет
    reasoning_effort: str | None    # что передавать в reasoning_effort ("none" — без рассуждений)
    note: str = ""
    vision: bool = True             # можно ли отправлять кадр (у некоторых моделей с кадром ломается формат ответа)


# Составлено по замерам docs/llm_benchmark.md (свой видео-материал, рассуждения выключены).
# Модели, которых здесь нет, идут после этих (не измерены) — по размеру.
DEFAULT_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        "gemma-4-26b-a4b", 1, "none",
        "быстрая (25 с/клип, с кадром 38 с), ровный формат в обоих режимах, самый низкий пик памяти, "
        "с кадром описывает реально видимое",
    ),
    ModelProfile(
        "qwen3.6-35b-a3b", 2, "none",
        "быстрая (28–37 с/клип) и яркие заголовки, но самая тяжёлая (20.6 ГиБ, загрузка ~105 с); с кадром "
        "ломает формат (хештегов 14–21 вместо 30, короткое описание) — кадр не отправляется",
        vision=False,
    ),
    ModelProfile(
        "qwen3.8-27b", 3, "none",
        "плотная 27B: 100+ с/клип (с кадром 180 с) и пик памяти до 17 ГиБ без выигрыша в качестве",
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
    pass


def read_mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo_path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable не найден в /proc/meminfo")


def estimate_footprint_mib(model: ModelInfo) -> float:
    size_mib = (model.size_bytes or 0) / MIB
    return size_mib * FOOTPRINT_RATIO + KV_CACHE_MIB


def _profile_for(key: str, profiles: tuple[ModelProfile, ...]) -> ModelProfile | None:
    lowered = key.lower()
    for profile in profiles:
        if profile.key_part.lower() in lowered:
            return profile
    return None


def default_reasoning_effort(model: ModelInfo) -> str | None:
    """Для модели вне рейтинга: рассуждения выключаем, если модель это позволяет
    (замеры: рассуждения увеличивают время клипа с секунд до минут без выигрыша
    в качестве коротких заголовков)."""
    return "none" if model.supports_reasoning_control else None


def available_after_unload_mib(models: list[ModelInfo], mem_available_mib: float) -> float:
    """MemAvailable + оценка памяти, которую вернёт выгрузка уже загруженных LLM."""
    return mem_available_mib + sum(estimate_footprint_mib(m) for m in models if m.loaded_instances)


def select_model(
    models: list[ModelInfo],
    available_mib: float,
    override: str | None = None,
    reserve_mib: float = DEFAULT_PIPELINE_RESERVE_MIB,
    profiles: tuple[ModelProfile, ...] | None = None,
) -> Selection:
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

    def sort_key(model: ModelInfo) -> tuple[int, float, float]:
        profile = _profile_for(model.key, profiles)
        if profile:
            return (0, profile.rank, 0.0)
        return (1, 0.0, -(model.size_bytes or 0))   # неизвестные — после известных, крупные раньше

    ranked = sorted(chat_models, key=sort_key)
    budget_mib = available_mib - reserve_mib
    skipped: list[str] = []

    for model in ranked:
        # available_mib — память, доступная, если выгрузить уже загруженные LLM (её считает
        # вызывающий код: MemAvailable + их занятое), поэтому нужная память всегда полная
        need_mib = estimate_footprint_mib(model)
        if need_mib <= budget_mib:
            profile = _profile_for(model.key, profiles)
            effort = profile.reasoning_effort if profile else default_reasoning_effort(model)
            place = f"место {ranked.index(model) + 1} из {len(ranked)}"
            reason = (
                f"{'уже загружена; ' if model.loaded_instances else ''}"
                f"{'лучшая в рейтинге' if not skipped else 'лучшая из помещающихся'} ({place}), "
                f"нужно ~{need_mib / 1024:.1f} ГиБ, доступно {available_mib / 1024:.1f} ГиБ "
                f"при резерве {reserve_mib / 1024:.1f} ГиБ под пайплайн"
            )
            if skipped:
                reason += "; не поместились: " + ", ".join(skipped)
            if excluded:
                reason += "; исключены не-чат модели: " + ", ".join(excluded)
            return Selection(
                model_key=model.key,
                reasoning_effort=effort,
                supports_vision=model.vision and (profile.vision if profile else True),
                reason=reason,
                already_loaded=bool(model.loaded_instances),
            )
        skipped.append(f"{model.key} (нужно ~{need_mib / 1024:.1f} ГиБ > {budget_mib / 1024:.1f})")

    raise NoSuitableModelError(
        f"Ни одна LLM не помещается в память: доступно {available_mib / 1024:.1f} ГиБ, "
        f"резерв под пайплайн {reserve_mib / 1024:.1f} ГиБ. " + "; ".join(skipped)
    )
