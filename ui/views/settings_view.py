"""Экран настроек (Функция 19): длина Shorts, стиль субтитров, стиль плашки,
положение логотипа, уровень AI Zoom, качество экспорта.

Работает напрямую с core.entities.settings.UserSettings (immutable dataclass) —
при любом изменении собирается новый UserSettings через with_field() и
эмитится наружу (settings_saved), где viewmodel сохранит его в SQLite через
database/repositories/settings_repository.py (Шаг 6-7).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from core.entities.settings import UserSettings
from ui.widgets.glass_panel import GlassPanel
from ui.widgets.update_panel import UpdatePanel

SUBTITLE_STYLE_PRESETS = ["modern_bold", "minimal_clean", "neon_pop", "classic_yellow"]
LOGO_POSITIONS = ["top_right", "top_left", "bottom_right", "bottom_left"]
BANNER_POSITIONS = ["bottom_center", "top_center", "bottom_left", "bottom_right"]
WEIGHT_FIELDS = [
    ("weight_motion_intensity", "Движение", "Насколько сильно меняется кадр"),
    ("weight_face_prominence", "Животное в кадре", "Доля кадров окна, где YOLO видит животное"),
    ("weight_scene_change", "Смены сцен", "Монтажные склейки внутри окна"),
    ("weight_audio_event", "Звуковые события", "Лай, мяуканье, мурлыканье, смех (YAMNet); нет модели — сигнал не считается"),
    ("weight_motion_events", "Прыжки и падения", "Бонус к оценке; не входит в нормировку остальных весов"),
]
QUALITY_PRESETS = ["low", "medium", "high"]
FPS_CHOICES = [24, 25, 30, 50, 60]
ENCODER_LABELS = {"auto": "Авто (VAAPI, если работает)", "vaapi": "VAAPI (видеокарта)", "x264": "libx264 (процессор)"}


class SettingsView(QWidget):
    settings_saved = Signal(object)  # UserSettings

    def __init__(
        self, initial_settings: UserSettings, parent: QWidget | None = None, available_effects: list[str] | None = None
    ) -> None:
        super().__init__(parent)
        self._settings = initial_settings

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(16)

        card = GlassPanel(self, corner_radius=16)
        form = QFormLayout(card)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # --- Длина Shorts ---
        self._durations_row = QWidget()
        durations_layout = QVBoxLayout(self._durations_row)
        durations_layout.setContentsMargins(0, 0, 0, 0)
        self._duration_checkboxes: dict[int, QCheckBox] = {}
        checkboxes_container = QWidget()
        from PySide6.QtWidgets import QHBoxLayout

        checkboxes_layout = QHBoxLayout(checkboxes_container)
        checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        for duration in (15, 20, 30, 35, 45, 60):
            checkbox = QCheckBox(f"{duration}с")
            checkbox.setChecked(duration in initial_settings.shorts.allowed_durations_sec)
            checkbox.stateChanged.connect(self._on_field_changed)
            self._duration_checkboxes[duration] = checkbox
            checkboxes_layout.addWidget(checkbox)
        durations_layout.addWidget(checkboxes_container)
        form.addRow(QLabel("Длина Shorts"), self._durations_row)

        # --- Viral Score Threshold ---
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(initial_settings.viral_score.queue_threshold)
        self._threshold_value_label = QLabel(str(initial_settings.viral_score.queue_threshold))
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_value_label.setText(str(v))
        )
        self._threshold_slider.valueChanged.connect(self._on_field_changed)
        threshold_row = QWidget()
        threshold_layout = self._build_slider_row(self._threshold_slider, self._threshold_value_label)
        form.addRow(QLabel("Порог Viral Score для очереди"), threshold_layout)

        # --- Веса Viral Score: по разложению оценки в карточке клипа видно, что дало очки ---
        self._weight_spins: dict[str, QDoubleSpinBox] = {}
        weights_container = QWidget()
        weights_layout = QFormLayout(weights_container)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.setSpacing(6)
        for field_name, caption, tooltip in WEIGHT_FIELDS:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(getattr(initial_settings.viral_score, field_name))
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(self._on_field_changed)
            self._weight_spins[field_name] = spin
            weights_layout.addRow(QLabel(caption), spin)
        form.addRow(QLabel("Веса Viral Score"), weights_container)

        # --- Стиль субтитров ---
        self._subtitle_style_combo = QComboBox()
        self._subtitle_style_combo.addItems(SUBTITLE_STYLE_PRESETS)
        self._subtitle_style_combo.setCurrentText(initial_settings.subtitles.style_preset)
        self._subtitle_style_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Стиль субтитров"), self._subtitle_style_combo)

        # --- Стиль/положение рекламной плашки ---
        self._banner_position_combo = QComboBox()
        self._banner_position_combo.addItems(BANNER_POSITIONS)
        self._banner_position_combo.setCurrentText(initial_settings.branding.banner_position)
        self._banner_position_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Положение рекламной плашки"), self._banner_position_combo)

        # --- Положение логотипа ---
        self._logo_position_combo = QComboBox()
        self._logo_position_combo.addItems(LOGO_POSITIONS)
        self._logo_position_combo.setCurrentText(initial_settings.branding.logo_position)
        self._logo_position_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Положение логотипа"), self._logo_position_combo)

        # --- AI Zoom ---
        self._ai_zoom_checkbox = QCheckBox("Включить AI Zoom")
        self._ai_zoom_checkbox.setChecked(initial_settings.reframe.ai_zoom_enabled)
        self._ai_zoom_checkbox.stateChanged.connect(self._on_field_changed)

        self._zoom_factor_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_factor_slider.setRange(10, 30)  # 1.0x - 3.0x, шаг 0.1
        self._zoom_factor_slider.setValue(int(initial_settings.reframe.max_zoom_factor * 10))
        self._zoom_factor_label = QLabel(f"{initial_settings.reframe.max_zoom_factor:.1f}x")
        self._zoom_factor_slider.valueChanged.connect(
            lambda v: self._zoom_factor_label.setText(f"{v / 10:.1f}x")
        )
        self._zoom_factor_slider.valueChanged.connect(self._on_field_changed)

        zoom_container = QWidget()
        zoom_layout = QVBoxLayout(zoom_container)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.addWidget(self._ai_zoom_checkbox)
        zoom_layout.addWidget(self._build_slider_row(self._zoom_factor_slider, self._zoom_factor_label))
        form.addRow(QLabel("AI Zoom"), zoom_container)

        # --- Безопасная зона Shorts (px при 1080x1920) ---
        self._safe_zone_spins: dict[str, QSpinBox] = {}
        safe_container = QWidget()
        from PySide6.QtWidgets import QHBoxLayout as _HBox

        safe_layout = _HBox(safe_container)
        safe_layout.setContentsMargins(0, 0, 0, 0)
        for field_name, caption, maximum in (
            ("top_px", "Верх", 600), ("bottom_px", "Низ", 800), ("left_px", "Лево", 300), ("right_px", "Право", 300),
        ):
            spin = QSpinBox()
            spin.setRange(0, maximum)
            spin.setSingleStep(10)
            spin.setSuffix(" px")
            spin.setValue(getattr(initial_settings.safe_zone, field_name))
            spin.valueChanged.connect(self._on_field_changed)
            self._safe_zone_spins[field_name] = spin
            safe_layout.addWidget(QLabel(caption))
            safe_layout.addWidget(spin)
        form.addRow(QLabel("Безопасная зона Shorts"), safe_container)

        # --- Качество экспорта ---
        self._quality_combo = QComboBox()
        self._quality_combo.addItems(QUALITY_PRESETS)
        self._quality_combo.setCurrentText(initial_settings.export.quality_preset)
        self._quality_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Качество экспорта"), self._quality_combo)

        # --- Фоновая музыка ---
        self._music_checkbox = QCheckBox("Фоновая музыка из assets/music/")
        self._music_checkbox.setChecked(initial_settings.audio.music_enabled)
        self._music_checkbox.stateChanged.connect(self._on_field_changed)
        self._music_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self._music_offset_slider.setRange(-30, 0)
        self._music_offset_slider.setValue(round(initial_settings.audio.music_offset_db))
        self._music_offset_slider.setToolTip("Музыка тише оригинального звука клипа на столько дБ (по громкости LUFS); на речи ещё тише")
        self._music_offset_label = QLabel(f"{self._music_offset_slider.value()} дБ")
        self._music_offset_slider.valueChanged.connect(lambda v: self._music_offset_label.setText(f"{v} дБ"))
        self._music_offset_slider.valueChanged.connect(self._on_field_changed)
        self._duck_checkbox = QCheckBox("Приглушать музыку на речи")
        self._duck_checkbox.setChecked(initial_settings.audio.duck_on_speech)
        self._duck_checkbox.stateChanged.connect(self._on_field_changed)
        music_container = QWidget()
        music_layout = QVBoxLayout(music_container)
        music_layout.setContentsMargins(0, 0, 0, 0)
        music_layout.addWidget(self._music_checkbox)
        music_layout.addWidget(self._build_slider_row(self._music_offset_slider, self._music_offset_label))
        music_layout.addWidget(self._duck_checkbox)
        form.addRow(QLabel("Музыка (дБ ниже оригинала)"), music_container)

        # --- Запас памяти под пайплайн (LLM берёт только то, что останется сверх него) ---
        self._reserve_spin = QDoubleSpinBox()
        self._reserve_spin.setRange(0.5, 24.0)
        self._reserve_spin.setSingleStep(0.5)
        self._reserve_spin.setSuffix(" ГиБ")
        self._reserve_spin.setValue(initial_settings.llm.pipeline_reserve_gb)
        self._reserve_spin.setToolTip(
            "Сколько памяти оставить Whisper, YOLO, ffmpeg и интерфейсу. Модель, которой не хватает места, не загружается: "
            "берётся меньшая или клипы получают заголовки по умолчанию (причина видна в очереди и в карточке клипа)."
        )
        self._reserve_spin.valueChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Запас памяти под пайплайн"), self._reserve_spin)

        # --- Очередь: сколько видео одновременно ---
        self._concurrent_spin = QSpinBox()
        self._concurrent_spin.setRange(1, 16)
        self._concurrent_spin.setValue(initial_settings.batch.max_concurrent_videos)
        self._concurrent_spin.setToolTip("Остальные видео ждут в очереди. Модели YOLO/Whisper/LLM общие на всю очередь.")
        self._concurrent_spin.valueChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Видео одновременно"), self._concurrent_spin)

        # --- Эффекты плагинов: какие применять к клипам (по порядку списка) ---
        self._effect_checkboxes: dict[str, QCheckBox] = {}
        effects_container = QWidget()
        effects_layout = QVBoxLayout(effects_container)
        effects_layout.setContentsMargins(0, 0, 0, 0)
        names = list(available_effects or [])
        for missing in initial_settings.plugins.enabled_effects:   # выбран раньше, а плагина сейчас нет — показываем, а не теряем молча
            if missing not in names:
                names.append(missing)
        for name in names:
            available = name in (available_effects or [])
            checkbox = QCheckBox(name if available else f"{name} (плагин не найден)")
            checkbox.setChecked(name in initial_settings.plugins.enabled_effects)
            checkbox.setEnabled(available or checkbox.isChecked())
            checkbox.stateChanged.connect(self._on_field_changed)
            self._effect_checkboxes[name] = checkbox
            effects_layout.addWidget(checkbox)
        if not names:
            hint = QLabel("Плагинов с эффектами нет: положите их в plugins/<имя>/plugin.py и перезапустите приложение")
            hint.setWordWrap(True)
            effects_layout.addWidget(hint)
        else:
            note = QLabel("Эффекты применяются к видеоряду клипа до субтитров и плашки; без выбранных эффектов экспорт не замедляется.")
            note.setWordWrap(True)
            effects_layout.addWidget(note)
        form.addRow(QLabel("Эффекты (плагины)"), effects_container)

        # --- FPS и битрейт клипа ---
        self._fps_combo = QComboBox()
        self._fps_combo.addItems([str(f) for f in FPS_CHOICES])
        self._fps_combo.setCurrentText(str(initial_settings.export.fps))
        if self._fps_combo.currentText() != str(initial_settings.export.fps):   # нестандартный fps из yaml/базы
            self._fps_combo.addItem(str(initial_settings.export.fps))
            self._fps_combo.setCurrentText(str(initial_settings.export.fps))
        self._fps_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Частота кадров (FPS)"), self._fps_combo)

        self._bitrate_spin = QSpinBox()
        self._bitrate_spin.setRange(1, 50)
        self._bitrate_spin.setSuffix(" Мбит/с")
        self._bitrate_spin.setValue(initial_settings.export.bitrate_mbps)
        self._bitrate_spin.setToolTip("Потолок видеобитрейта клипа (вместе с качеством экспорта: кадр не хуже пресета, поток не выше этого)")
        self._bitrate_spin.valueChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Битрейт видео (максимум)"), self._bitrate_spin)

        self._encoder_combo = QComboBox()
        for value, label in ENCODER_LABELS.items():
            self._encoder_combo.addItem(label, value)
        index = self._encoder_combo.findData(initial_settings.export.encoder)
        self._encoder_combo.setCurrentIndex(max(0, index))
        self._encoder_combo.setToolTip(
            "Аппаратное кодирование (VAAPI на Radeon) втрое-вчетверо быстрее, файл при том же качестве немного больше. "
            "Нужен mesa-va-drivers-freeworld (ставится scripts/setup.sh); если не работает — автоматически libx264."
        )
        self._encoder_combo.currentIndexChanged.connect(self._on_field_changed)
        form.addRow(QLabel("Видеокодер"), self._encoder_combo)

        # --- Обновления: только по кнопке ---
        self.update_panel = UpdatePanel(Path(__file__).resolve().parents[2])
        form.addRow(QLabel("Обновления"), self.update_panel)

        self._save_button = QPushButton("Сохранить настройки")
        self._save_button.setObjectName("primaryButton")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_clicked)

        root_layout.addWidget(card)
        root_layout.addWidget(self._save_button)
        root_layout.addStretch()

    @staticmethod
    def _build_slider_row(slider: QSlider, value_label: QLabel) -> QWidget:
        from PySide6.QtWidgets import QHBoxLayout

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        return row

    def _on_field_changed(self, *_args: object) -> None:
        self._save_button.setEnabled(True)

    def _on_save_clicked(self) -> None:
        selected_durations = tuple(
            sorted(d for d, cb in self._duration_checkboxes.items() if cb.isChecked())
        )
        updated = self._settings.with_field(
            "shorts", allowed_durations_sec=selected_durations or self._settings.shorts.allowed_durations_sec
        )
        updated = updated.with_field(
            "viral_score", queue_threshold=self._threshold_slider.value(),
            **{name: round(spin.value(), 2) for name, spin in self._weight_spins.items()},
        )
        updated = updated.with_field("subtitles", style_preset=self._subtitle_style_combo.currentText())
        updated = updated.with_field(
            "branding",
            banner_position=self._banner_position_combo.currentText(),
            logo_position=self._logo_position_combo.currentText(),
        )
        updated = updated.with_field(
            "reframe",
            ai_zoom_enabled=self._ai_zoom_checkbox.isChecked(),
            max_zoom_factor=self._zoom_factor_slider.value() / 10,
        )
        updated = updated.with_field(
            "export", quality_preset=self._quality_combo.currentText(), fps=int(self._fps_combo.currentText()),
            bitrate_mbps=self._bitrate_spin.value(), encoder=self._encoder_combo.currentData(),
        )
        updated = updated.with_field(
            "safe_zone", **{name: spin.value() for name, spin in self._safe_zone_spins.items()}
        )

        updated = updated.with_field("plugins", enabled_effects=tuple(n for n, cb in self._effect_checkboxes.items() if cb.isChecked()))
        updated = updated.with_field("llm", pipeline_reserve_gb=self._reserve_spin.value())
        updated = updated.with_field("batch", max_concurrent_videos=self._concurrent_spin.value())
        updated = updated.with_field("audio", music_enabled=self._music_checkbox.isChecked(), music_offset_db=float(self._music_offset_slider.value()), duck_on_speech=self._duck_checkbox.isChecked())

        self._settings = updated
        self._save_button.setEnabled(False)
        self.settings_saved.emit(updated)
