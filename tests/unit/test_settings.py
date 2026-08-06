from pathlib import Path

from core.entities.settings import UserSettings

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default_settings.yaml"


def test_load_from_yaml_reads_real_config():
    settings = UserSettings.load_from_yaml(CONFIG_PATH)
    assert settings.shorts.allowed_durations_sec == (15, 20, 30, 35, 45, 60)
    assert settings.export.width == 1080
    assert settings.export.height == 1920


def test_with_field_returns_new_immutable_object():
    settings = UserSettings.load_from_yaml(CONFIG_PATH)
    updated = settings.with_field("export", quality_preset="medium")

    assert updated.export.quality_preset == "medium"
    assert settings.export.quality_preset != "medium"  # оригинал не мутирован
