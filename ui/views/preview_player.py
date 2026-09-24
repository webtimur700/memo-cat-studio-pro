"""Плеер предпросмотра — реальное воспроизведение через QMediaPlayer/QVideoWidget
(Qt Multimedia, бэкенд FFmpeg — тот же FFmpeg, что используется в video/ffmpeg_wrapper.py,
поэтому предпросмотр и финальный экспорт декодируют файл одинаково).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


def _format_ms(ms: int) -> str:
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


class PreviewPlayer(QWidget):
    _source_requested = Signal(QUrl)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._video_widget = QVideoWidget(self)
        self._video_widget.setMinimumHeight(360)
        self._video_widget.setStyleSheet("background-color: #000000; border-radius: 12px;")

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoOutput(self._video_widget)
        # Смена клипа при уже открытом другом: QMediaPlayer.setSource ждёт остановки потока декодера,
        # а тот при разрушении своих объектов берёт GIL, который держит вызвавший Python-код — интерфейс
        # замирает навсегда (воспроизводится на реальных 1080x1920 h264). Поэтому setSource вызывается
        # НАТИВНОЙ queued-связью прямо из цикла событий, без Python-кадров и с отпущенным GIL.
        self._requested_path: Path | None = None
        self._source_requested.connect(self._player.setSource, Qt.ConnectionType.QueuedConnection)

        self._play_button = QPushButton("▶")
        self._play_button.setFixedWidth(40)
        self._play_button.clicked.connect(self._toggle_play_pause)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.sliderMoved.connect(self._player.setPosition)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setStyleSheet("color: #96969a;")

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._pause_when_loaded = False

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self._play_button)
        controls_layout.addWidget(self._position_slider, stretch=1)
        controls_layout.addWidget(self._time_label)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._video_widget, stretch=1)
        root_layout.addLayout(controls_layout)

    @property
    def source_path(self) -> Path | None:
        return self._requested_path   # сам QMediaPlayer получает источник чуть позже (queued setSource)

    def load_video(self, path: Path, autoplay: bool = False) -> None:
        """Загружает клип; pause() показывает первый кадр вместо чёрного экрана."""
        self._pause_when_loaded = not autoplay
        self._requested_path = path
        self._source_requested.emit(QUrl.fromLocalFile(str(path)))
        if autoplay:
            QTimer.singleShot(0, self._player.play)   # после queued setSource

    def play(self) -> None:
        self._player.play()

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        # pause() до окончания загрузки ничего не показывает; после — выводит первый кадр
        if self._pause_when_loaded and status in (
            QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia
        ):
            self._pause_when_loaded = False
            # pause() прямо из обработчика mediaStatusChanged (он вызывается синхронно из setSource)
            # вешает Qt Multimedia (ffmpeg-бэкенд) намертво, когда клип меняют при уже открытом
            # другом: интерфейс замирает. Откладываем в цикл событий.
            QTimer.singleShot(0, self._player.pause)

    def seek_to(self, seconds: float) -> None:
        self._player.setPosition(int(seconds * 1000))
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self._player.play()

    def _toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self._play_button.setText("⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

    def _on_position_changed(self, position_ms: int) -> None:
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(position_ms)
        self._position_slider.blockSignals(False)
        self._time_label.setText(f"{_format_ms(position_ms)} / {_format_ms(self._player.duration())}")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._position_slider.setRange(0, duration_ms)
        self._time_label.setText(f"{_format_ms(self._player.position())} / {_format_ms(duration_ms)}")
