from pathlib import Path

import numpy as np

from plugins.loader import load_plugins
from plugins.sdk.registry import PluginRegistry

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "plugins" / "examples"


def test_example_plugin_loads_and_registers():
    registry = PluginRegistry()
    loaded = load_plugins(EXAMPLES_DIR, registry)

    assert len(loaded) == 1
    assert loaded[0].name == "vintage_sepia"
    assert "vintage_sepia" in registry.list_effects()


def test_registered_effect_actually_transforms_frame():
    registry = PluginRegistry()
    load_plugins(EXAMPLES_DIR, registry)
    effect = registry.get_effect("vintage_sepia")

    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = effect(frame)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, frame)


def test_duplicate_registration_raises():
    registry = PluginRegistry()
    registry.register_effect("foo", lambda frame: frame)
    try:
        registry.register_effect("foo", lambda frame: frame)
        assert False, "должно было бросить ValueError"
    except ValueError:
        pass


def test_broken_plugin_does_not_break_others(tmp_path: Path):
    good_dir = tmp_path / "good_plugin"
    good_dir.mkdir()
    (good_dir / "plugin.py").write_text((EXAMPLES_DIR / "example_effect_plugin" / "plugin.py").read_text())

    broken_dir = tmp_path / "broken_plugin"
    broken_dir.mkdir()
    (broken_dir / "plugin.py").write_text("raise RuntimeError('специально сломан для теста')")

    registry = PluginRegistry()
    loaded = load_plugins(tmp_path, registry)

    assert len(loaded) == 1
    assert loaded[0].name == "vintage_sepia"
