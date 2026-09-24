"""Доменная модель пользовательских настроек (Функция 19).

Это единственное место, откуда UI и Application-сервисы читают настройки —
и UI (settings_view.py), и scoring/cutting/export сервисы работают с одним
и тем же объектом, а не с разрозненными dict'ами.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ShortsSettings:
    allowed_durations_sec: tuple[int, ...] = (15, 20, 30, 35, 45, 60)
    min_duration_sec: int = 15
    max_duration_sec: int = 60
    boundary_padding_ms: int = 250
    max_moments: int = 10
    max_coverage_ratio: float = 0.35     # клипы вместе — не больше этой доли исходного видео
    snap_tolerance_sec: float = 6.0      # на сколько границу можно сдвинуть к паузе речи / смене сцены


@dataclass(frozen=True, slots=True)
class ViralScoreSettings:
    queue_threshold: int = 55            # абсолютный минимум: окна ниже него не берутся никогда
    relative_top_ratio: float = 0.40     # и берутся только лучшие 40% окон САМОГО видео
    weight_motion_intensity: float = 0.30
    weight_scene_change: float = 0.15
    weight_audio_event: float = 0.25
    weight_face_prominence: float = 0.15
    weight_speech_presence: float = 0.15


@dataclass(frozen=True, slots=True)
class ReframeSettings:
    target_width: int = 1080
    target_height: int = 1920
    tracking_smoothing_alpha: float = 0.15
    ai_zoom_enabled: bool = True
    min_subject_area_ratio: float = 0.08
    max_zoom_factor: float = 2.2


@dataclass(frozen=True, slots=True)
class SubtitleSettings:
    model_size: str = "small"
    compute_type: str = "int8"
    style_preset: str = "modern_bold"
    burn_in: bool = True
    word_highlight: bool = True
    language: str = "auto"       # язык речи: "auto" — определять по звуку (принудительный язык на чужой речи даёт мусор)
    translate_to: str = "ru"     # переводить субтитры через LLM на этот язык ("" — не переводить)


@dataclass(frozen=True, slots=True)
class BrandingSettings:
    banner_position: str = "bottom_center"
    logo_position: str = "top_right"
    subscribe_button_enabled: bool = True
    collision_avoidance: bool = True
    banner_text_lines: tuple[str, ...] = (
        "💼 100zarplat.ru",
        "Подбор вакансий от проверенных работодателей",
        "💰 Ежедневные и еженедельные выплаты",
    )


@dataclass(frozen=True, slots=True)
class SafeZoneSettings:
    """Поля (px при 1080x1920), которые интерфейс YouTube Shorts перекрывает:
    сверху — поиск/камера, снизу — название, канал, описание, справа — колонка
    лайков/комментариев. Плашка, субтитры, логотип и Subscribe ставятся внутри."""

    top_px: int = 250
    bottom_px: int = 450
    left_px: int = 60
    right_px: int = 150


@dataclass(frozen=True, slots=True)
class ExportSettings:
    codec_video: str = "h264"
    codec_audio: str = "aac"
    fps: int = 30
    width: int = 1080
    height: int = 1920
    quality_preset: str = "high"
    bitrate_mbps: int = 12


@dataclass(frozen=True, slots=True)
class BatchSettings:
    """Очередь пакетной обработки: сколько видео обрабатывается одновременно (остальные ждут)."""

    max_concurrent_videos: int = 1

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "BatchSettings":
        return cls(max_concurrent_videos=max(1, int(raw.get("max_concurrent_videos", 1))))


@dataclass(frozen=True, slots=True)
class UserSettings:
    """Корневой агрегат настроек. Immutable — изменения выполняются через
    replace_field(), что предотвращает случайную мутацию shared-объекта
    из нескольких потоков worker'ов одновременно.
    """

    shorts: ShortsSettings = field(default_factory=ShortsSettings)
    viral_score: ViralScoreSettings = field(default_factory=ViralScoreSettings)
    reframe: ReframeSettings = field(default_factory=ReframeSettings)
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)
    branding: BrandingSettings = field(default_factory=BrandingSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    safe_zone: SafeZoneSettings = field(default_factory=SafeZoneSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)

    @classmethod
    def load_from_yaml(cls, path: Path) -> "UserSettings":
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        shorts_raw = raw.get("shorts", {})
        viral_raw = raw.get("viral_score", {})
        weights_raw = viral_raw.get("weights", {})
        reframe_raw = raw.get("reframe", {})
        zoom_raw = reframe_raw.get("ai_zoom", {})
        subs_raw = raw.get("subtitles", {})
        branding_raw = raw.get("branding", {})
        export_raw = raw.get("export", {})
        safe_raw = raw.get("safe_zone", {})

        return cls(
            shorts=ShortsSettings(
                allowed_durations_sec=tuple(
                    shorts_raw.get("allowed_durations_sec", (15, 20, 30, 35, 45, 60))
                ),
                min_duration_sec=shorts_raw.get("min_duration_sec", 15),
                max_duration_sec=shorts_raw.get("max_duration_sec", 60),
                boundary_padding_ms=shorts_raw.get("boundary_padding_ms", 250),
                max_moments=shorts_raw.get("max_moments", 10),
                max_coverage_ratio=shorts_raw.get("max_coverage_ratio", 0.35),
                snap_tolerance_sec=shorts_raw.get("snap_tolerance_sec", 6.0),
            ),
            viral_score=ViralScoreSettings(
                queue_threshold=viral_raw.get("queue_threshold", 55),
                relative_top_ratio=viral_raw.get("relative_top_ratio", 0.40),
                weight_motion_intensity=weights_raw.get("motion_intensity", 0.30),
                weight_scene_change=weights_raw.get("scene_change", 0.15),
                weight_audio_event=weights_raw.get("audio_event", 0.25),
                weight_face_prominence=weights_raw.get("face_prominence", 0.15),
                weight_speech_presence=weights_raw.get("speech_presence", 0.15),
            ),
            reframe=ReframeSettings(
                target_width=reframe_raw.get("target_resolution", [1080, 1920])[0],
                target_height=reframe_raw.get("target_resolution", [1080, 1920])[1],
                tracking_smoothing_alpha=reframe_raw.get("tracking_smoothing_alpha", 0.15),
                ai_zoom_enabled=zoom_raw.get("enabled", True),
                min_subject_area_ratio=zoom_raw.get("min_subject_area_ratio", 0.08),
                max_zoom_factor=zoom_raw.get("max_zoom_factor", 2.2),
            ),
            subtitles=SubtitleSettings(
                model_size=subs_raw.get("model_size", "small"),
                compute_type=subs_raw.get("compute_type", "int8"),
                style_preset=subs_raw.get("style_preset", "modern_bold"),
                burn_in=subs_raw.get("burn_in", True),
                word_highlight=subs_raw.get("word_highlight", True),
                language=subs_raw.get("language", "auto"),
                translate_to=subs_raw.get("translate_to", "ru"),
            ),
            branding=BrandingSettings(
                banner_position=branding_raw.get("banner", {}).get("position", "bottom_center"),
                logo_position=branding_raw.get("logo", {}).get("position", "top_right"),
                subscribe_button_enabled=branding_raw.get("subscribe_button", {}).get(
                    "enabled", True
                ),
                collision_avoidance=branding_raw.get("banner", {}).get(
                    "collision_avoidance", True
                ),
                banner_text_lines=tuple(
                    branding_raw.get("banner", {}).get(
                        "text_lines",
                        [
                            "💼 100zarplat.ru",
                            "Подбор вакансий от проверенных работодателей",
                            "💰 Ежедневные и еженедельные выплаты",
                        ],
                    )
                ),
            ),
            export=ExportSettings(
                codec_video=export_raw.get("codec_video", "h264"),
                codec_audio=export_raw.get("codec_audio", "aac"),
                fps=export_raw.get("fps", 30),
                width=export_raw.get("resolution", [1080, 1920])[0],
                height=export_raw.get("resolution", [1080, 1920])[1],
                quality_preset=export_raw.get("quality_preset", "high"),
                bitrate_mbps=export_raw.get("bitrate_mbps", 12),
            ),
            safe_zone=SafeZoneSettings(
                top_px=safe_raw.get("top_px", 250),
                bottom_px=safe_raw.get("bottom_px", 450),
                left_px=safe_raw.get("left_px", 60),
                right_px=safe_raw.get("right_px", 150),
            ),
            batch=BatchSettings.from_raw(raw.get("batch", {})),
        )

    def with_field(self, section: str, **changes: Any) -> "UserSettings":
        """Возвращает новый UserSettings с изменённым вложенным разделом.

        Пример: settings.with_field("export", quality_preset="medium")
        """
        current_section = getattr(self, section)
        updated_section = replace(current_section, **changes)
        return replace(self, **{section: updated_section})
