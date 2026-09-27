import pytest

from llm.lm_studio_api import GIB, ModelInfo
from llm.model_selector import (
    ModelProfile,
    NoSuitableModelError,
    available_after_unload_mib,
    estimate_footprint_mib,
    read_mem_available_mib,
    select_model,
)


def _model(key, gib, kind="llm", vision=True, reasoning=("off", "on"), loaded=()):
    return ModelInfo(key=key, kind=kind, size_bytes=int(gib * GIB), vision=vision,
                     reasoning_options=tuple(reasoning), loaded_instances=tuple(loaded))


MODELS = [
    _model("text-embedding-nomic-embed-text-v1.5", 0.08, kind="embedding", vision=False, reasoning=()),
    _model("qwen/qwen3.8-27b", 16.5),
    _model("google/gemma-4-26b-a4b", 16.8),
    _model("qwen/qwen3.6-35b-a3b", 20.6),
]
PROFILES = (
    ModelProfile("gemma-4-26b", 1, "none"),
    ModelProfile("qwen3.6", 2, "none"),
    ModelProfile("qwen3.8", 3, "none"),
)
BIG = 28 * 1024   # МиБ


def test_embedding_never_chosen_even_if_first_in_list():
    models = [MODELS[0]] + MODELS[1:]
    selection = select_model(models, BIG, profiles=PROFILES)
    assert "embed" not in selection.model_key
    assert "text-embedding-nomic-embed-text-v1.5" in selection.reason     # исключение отражено в причине


def test_best_ranked_model_wins_when_memory_is_plentiful():
    selection = select_model(MODELS, BIG, profiles=PROFILES)
    assert selection.model_key == "google/gemma-4-26b-a4b"
    assert selection.reasoning_effort == "none" and selection.supports_vision
    assert "лучшая в рейтинге" in selection.reason


def test_smaller_model_taken_when_best_does_not_fit():
    small = _model("tiny/model-4b", 2.5)
    # доступно чуть больше запаса + маленькая модель, крупные не влезают
    available = 6 * 1024 + estimate_footprint_mib(small, PROFILES) + 200
    selection = select_model(MODELS + [small], available, profiles=PROFILES)
    assert selection.model_key == "tiny/model-4b"
    assert "не поместились" in selection.reason and "лучшая из помещающихся" in selection.reason


def test_next_ranked_taken_when_only_first_is_too_big():
    profiles = (ModelProfile("qwen3.6", 1, "none"), ModelProfile("gemma-4-26b", 2, "none"))
    # qwen3.6 (20.6 ГиБ) не влезает, gemma (16.8) влезает
    available = 6 * 1024 + estimate_footprint_mib(MODELS[2], profiles) + 100
    assert select_model(MODELS, available, profiles=profiles).model_key == "google/gemma-4-26b-a4b"


def test_override_has_priority_and_skips_memory_check():
    selection = select_model(MODELS, available_mib=100, override="qwen/qwen3.8-27b", profiles=PROFILES)
    assert selection.model_key == "qwen/qwen3.8-27b" and selection.from_override
    assert "LM_STUDIO_MODEL_OVERRIDE" in selection.reason


def test_unknown_models_after_known_larger_first():
    unknown_small, unknown_big = _model("x/small-7b", 4.4), _model("x/big-14b", 8.4)
    selection = select_model([unknown_small, unknown_big], BIG, profiles=())
    assert selection.model_key == "x/big-14b"                  # без рейтинга — крупнее лучше, если влезает
    mixed = select_model([unknown_big, MODELS[2]], BIG, profiles=PROFILES)
    assert mixed.model_key == "google/gemma-4-26b-a4b"          # известная раньше неизвестной


def test_only_embedding_models_is_a_clear_error():
    with pytest.raises(NoSuitableModelError, match="нет чат-моделей"):
        select_model([MODELS[0]], BIG, profiles=PROFILES)


def test_nothing_fits_is_a_clear_error():
    with pytest.raises(NoSuitableModelError, match="не помещается"):
        select_model(MODELS, available_mib=7 * 1024, profiles=PROFILES)


def test_model_without_reasoning_control_gets_no_effort():
    plain = _model("qwen/qwen2.5-7b-instruct", 4.4, vision=False, reasoning=())
    selection = select_model([plain], BIG, profiles=())
    assert selection.reasoning_effort is None and not selection.supports_vision


def test_loaded_model_memory_is_returned_by_unload_estimate():
    from llm.model_selector import DEFAULT_PROFILES, UNLOAD_RETURN_RATIO

    loaded = [_model("google/gemma-4-26b-a4b", 16.8, loaded=("google/gemma-4-26b-a4b",))] + MODELS[:1]
    gemma = next(p for p in DEFAULT_PROFILES if p.key_part == "gemma-4-26b-a4b")
    assert available_after_unload_mib(loaded, 1000) == pytest.approx(1000 + gemma.loaded_mib)   # измеренное
    unknown = [_model("x/model-7b", 4.0, loaded=("x/model-7b",))]
    assert available_after_unload_mib(unknown, 1000) == pytest.approx(1000 + 4.0 * 1024 * UNLOAD_RETURN_RATIO)


def test_mem_available_parsed(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 32130528 kB\nMemAvailable:   24058096 kB\n")
    assert read_mem_available_mib(meminfo) == pytest.approx(24058096 / 1024)


def test_real_profiles_rank_measured_models_and_switch_qwen36_vision_off():
    from llm.model_selector import DEFAULT_PROFILES

    ranked = [p.key_part for p in sorted(DEFAULT_PROFILES, key=lambda p: p.rank)]
    assert ranked == ["gemma-4-26b-a4b", "qwen3.6-35b-a3b", "qwen3.8-27b"]
    assert all(p.reasoning_effort == "none" for p in DEFAULT_PROFILES)

    only_qwen36 = select_model([MODELS[3]], BIG, profiles=DEFAULT_PROFILES)
    assert only_qwen36.model_key == "qwen/qwen3.6-35b-a3b" and not only_qwen36.supports_vision
    gemma = select_model(MODELS, BIG, profiles=DEFAULT_PROFILES)
    assert gemma.model_key == "google/gemma-4-26b-a4b" and gemma.supports_vision


def test_real_profiles_pick_gemma_on_the_users_machine_state():
    """Реальная ситуация: ~24 ГиБ доступно, в списке (в этом порядке) все скачанные модели."""
    from llm.model_selector import DEFAULT_PROFILES

    models = [MODELS[1], MODELS[2], MODELS[3], MODELS[0]]     # порядок как у /v1/models: qwen3.8, gemma, qwen3.6, embedding
    selection = select_model(models, available_mib=24 * 1024, profiles=DEFAULT_PROFILES)
    assert selection.model_key == "google/gemma-4-26b-a4b"
    tight = select_model(models, available_mib=21 * 1024, profiles=DEFAULT_PROFILES)
    assert tight.model_key == "google/gemma-4-26b-a4b"
    with pytest.raises(NoSuitableModelError):
        select_model(models, available_mib=12 * 1024, profiles=DEFAULT_PROFILES)


def test_small_unranked_backup_model_is_used_when_the_ranked_ones_do_not_fit():
    from llm.model_selector import DEFAULT_PROFILES

    backup = _model("someone/tiny-instruct-3b", 2.0, vision=False, reasoning=())
    selection = select_model(MODELS + [backup], available_mib=9 * 1024, profiles=DEFAULT_PROFILES)   # резерв 6 -> бюджет 3 ГиБ
    assert selection.model_key == "someone/tiny-instruct-3b"
    assert "запасная вне рейтинга" in selection.reason and "не поместились" in selection.reason
    # а когда лучшая помещается, запасная не трогается
    assert select_model(MODELS + [backup], available_mib=24 * 1024, profiles=DEFAULT_PROFILES).model_key == "google/gemma-4-26b-a4b"


def test_no_memory_error_carries_the_numbers_of_the_lightest_model():
    with pytest.raises(NoSuitableModelError) as info:
        select_model(MODELS, available_mib=10 * 1024, reserve_mib=6 * 1024, profiles=())
    err = info.value
    assert err.lightest_key == "qwen/qwen3.8-27b"
    assert err.need_mib == pytest.approx(estimate_footprint_mib(MODELS[1], ()))
    assert err.available_mib == 10 * 1024 and err.reserve_mib == 6 * 1024


def test_smaller_reserve_lets_a_model_fit():
    with pytest.raises(NoSuitableModelError):
        select_model(MODELS, available_mib=20 * 1024, reserve_mib=6 * 1024, profiles=PROFILES)
    assert select_model(MODELS, available_mib=20 * 1024, reserve_mib=2 * 1024, profiles=PROFILES).model_key == "google/gemma-4-26b-a4b"


# ---------------------------------------------------------------- запасная модель: меньше и не медленнее

def _p(key, gib, params):
    return ModelInfo(key=key, size_bytes=int(gib * GIB), params=params, vision=True, reasoning_options=("off", "on"))


REAL = [   # как в LM Studio на машине пользователя (размеры и params_string из /api/v1/models)
    _p("qwen/qwen3.8-27b", 16.52, "27B"),
    _p("google/gemma-4-26b-a4b", 16.76, "26B-A4B"),
    _p("qwen/qwen3.6-35b-a3b", 20.55, "35B-A3B"),
    _model("text-embedding-nomic-embed-text-v1.5", 0.08, kind="embedding", vision=False, reasoning=()),
]


def test_the_2026_09_26_case_now_picks_gemma_not_the_dense_qwen():
    """Было: доступно 20.2 ГиБ, резерв 6 — Gemma «не влезла на 0.2 ГиБ» (оценка по файлу 14.4), взята Qwen3.8 (14.2).
    По замерам пик Qwen3.8 выше, чем у Gemma, поэтому по файлу оценка шла в обратную сторону."""
    from llm.model_selector import DEFAULT_PROFILES

    selection = select_model(REAL, available_mib=20.2 * 1024, reserve_mib=6 * 1024, profiles=DEFAULT_PROFILES)
    assert selection.model_key == "google/gemma-4-26b-a4b"
    assert estimate_footprint_mib(REAL[0]) > estimate_footprint_mib(REAL[1])   # плотная 27B тяжелее MoE 26B-A4B


def test_slower_model_is_never_a_fallback_even_if_it_fits():
    profiles = (
        ModelProfile("gemma-4-26b", 1, "none", peak_mib=15 * 1024, sec_per_clip=30),
        ModelProfile("qwen3.8", 3, "none", peak_mib=10 * 1024, sec_per_clip=170),
    )
    models = [REAL[0], REAL[1]]
    with pytest.raises(NoSuitableModelError) as info:
        select_model(models, available_mib=18 * 1024, reserve_mib=6 * 1024, profiles=profiles)   # бюджет 12
    assert "qwen/qwen3.8-27b" in str(info.value) and "медленнее" in str(info.value)
    assert info.value.lightest_key == "google/gemma-4-26b-a4b"        # цифры в интерфейсе — про Gemma, а не про Qwen3.8
    assert select_model(models, available_mib=22 * 1024, profiles=profiles).model_key == "google/gemma-4-26b-a4b"


def test_smaller_faster_unmeasured_model_is_a_fallback_but_a_dense_one_is_not():
    from llm.model_selector import DEFAULT_PROFILES

    small_moe = _p("x/tiny-moe", 3.0, "8B-A1B")
    small_dense = _p("x/dense-14b", 8.0, "14B")
    selection = select_model(REAL + [small_dense, small_moe], available_mib=16 * 1024, profiles=DEFAULT_PROFILES)
    assert selection.model_key == "x/tiny-moe"                       # 1B активных — не медленнее Gemma (4B)
    assert "x/dense-14b" in selection.reason and "медленнее" in selection.reason


def test_active_params_parsing():
    from llm.model_selector import active_params_b

    assert active_params_b(_p("a", 1, "26B-A4B")) == 4
    assert active_params_b(_p("a", 1, "35B-A3B")) == 3
    assert active_params_b(_p("a", 1, "27B")) == 27
    assert active_params_b(_p("a", 1, "7.6B")) == 7.6
    assert active_params_b(_p("a", 1, None)) is None


def test_model_that_timed_out_goes_after_the_others_but_is_not_banned():
    from llm.model_selector import DEFAULT_PROFILES

    demoted = frozenset({"google/gemma-4-26b-a4b"})
    selection = select_model(REAL, available_mib=28 * 1024, profiles=DEFAULT_PROFILES, penalized=demoted)
    assert selection.model_key == "qwen/qwen3.6-35b-a3b"
    assert "понижены за недавний таймаут: google/gemma-4-26b-a4b" in selection.reason
    # Qwen3.6 не влезает — Gemma всё ещё лучше, чем ничего (меньше и не медленнее), Qwen3.8 — медленнее
    tight = select_model(REAL, available_mib=19 * 1024, profiles=DEFAULT_PROFILES, penalized=demoted)
    assert tight.model_key == "google/gemma-4-26b-a4b"


def test_already_loaded_heavy_model_gets_no_free_memory():
    """Qwen3.8 уже загружена: занятое ею возвращается в доступное, но нужна ей всё та же полная память —
    она не выигрывает у Gemma из-за того, что уже в памяти."""
    from llm.model_selector import DEFAULT_PROFILES

    loaded = [
        ModelInfo(key="qwen/qwen3.8-27b", size_bytes=int(16.52 * GIB), params="27B", vision=True,
                  reasoning_options=("off",), loaded_instances=("qwen/qwen3.8-27b",)),
        REAL[1], REAL[2],
    ]
    mem_available = 10.5 * 1024      # Qwen3.8 уже съела ~10 ГиБ
    available = available_after_unload_mib(loaded, mem_available, DEFAULT_PROFILES)
    selection = select_model(loaded, available_mib=available, profiles=DEFAULT_PROFILES)
    assert selection.model_key == "google/gemma-4-26b-a4b" and not selection.already_loaded
