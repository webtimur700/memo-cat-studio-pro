"""Панель деталей клипа: 10 заголовков, описание, хештеги — с кнопками
копирования, и кнопка «Открыть папку»."""

from __future__ import annotations

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.entities.llm_issue import LLMIssue
from ui.viewmodels.clip_results import ClipResult
from ui.widgets.score_breakdown import ScoreBreakdownWidget

COPIED_FEEDBACK_MS = 1200


def _copy_button(text_getter, label: str = "Копировать") -> QPushButton:
    button = QPushButton(label)
    button.setObjectName("copyButton")

    def on_click() -> None:
        QApplication.clipboard().setText(text_getter())
        button.setText("Скопировано ✓")
        QTimer.singleShot(COPIED_FEEDBACK_MS, lambda: button.setText(label))

    button.clicked.connect(on_click)
    return button


class ClipDetailsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: ClipResult | None = None

        self._header = QLabel("Выберите клип в списке")
        self._header.setObjectName("detailsHeader")
        self._header.setWordWrap(True)

        self._open_folder_button = QPushButton("Открыть папку")
        self._open_folder_button.setObjectName("primaryButton")
        self._open_folder_button.clicked.connect(self.open_folder)
        self._copy_all_button = _copy_button(lambda: self._result.full_text() if self._result else "", "Копировать всё")

        top = QHBoxLayout()
        top.addWidget(self._open_folder_button)
        top.addWidget(self._copy_all_button)

        self._titles_box = QVBoxLayout()
        self._titles_box.setSpacing(6)
        self._titles_container = QWidget()
        self._titles_container.setLayout(self._titles_box)

        self._description = QPlainTextEdit()
        self._description.setReadOnly(True)
        self._description.setFixedHeight(110)
        self._description_copy = _copy_button(lambda: self._description.toPlainText())

        self._hashtags = QPlainTextEdit()
        self._hashtags.setReadOnly(True)
        self._hashtags.setFixedHeight(90)
        self._hashtags_copy = _copy_button(lambda: self._hashtags.toPlainText())

        self._transcript = QLabel()
        self._transcript.setWordWrap(True)
        self._transcript.setObjectName("clipMeta")

        self._llm_note = QLabel()
        self._llm_note.setWordWrap(True)
        self._llm_note.setObjectName("llmWarning")
        self._llm_note.setStyleSheet("color: #f0b429; background: rgba(240,180,41,0.10); border-radius: 8px; padding: 8px;")
        self._llm_note.hide()

        self._breakdown = ScoreBreakdownWidget()

        content = QWidget()
        content.setObjectName("detailsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(self._header)
        layout.addLayout(top)
        layout.addWidget(self._section_label("Из чего сложился Viral Score"))
        layout.addWidget(self._breakdown)
        layout.addWidget(self._section_label("Заголовки"))
        layout.addWidget(self._titles_container)
        layout.addWidget(self._llm_note)
        layout.addLayout(self._section_row("Описание", self._description_copy))
        layout.addWidget(self._description)
        layout.addLayout(self._section_row("Хештеги", self._hashtags_copy))
        layout.addWidget(self._hashtags)
        layout.addWidget(self._section_label("Что говорят в клипе"))
        layout.addWidget(self._transcript)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("detailsScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        self._set_enabled(False)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _section_row(self, text: str, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self._section_label(text))
        row.addStretch()
        row.addWidget(button)
        return row

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self._open_folder_button, self._copy_all_button, self._description_copy, self._hashtags_copy):
            widget.setEnabled(enabled)

    def _clear_titles(self) -> None:
        """Убирает строки заголовков; виджеты отвязываются сразу (а не только по deleteLater)."""

        def drop(layout) -> None:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
                elif item.layout():
                    drop(item.layout())

        drop(self._titles_box)

    def set_clip(self, result: ClipResult | None) -> None:
        self._result = result
        self._clear_titles()

        if result is None:
            self._header.setText("Выберите клип в списке")
            self._description.clear()
            self._hashtags.clear()
            self._transcript.clear()
            self._llm_note.clear()
            self._llm_note.hide()
            self._breakdown.set_breakdown(0, ())
            self._set_enabled(False)
            return

        self._set_enabled(True)
        self._breakdown.set_breakdown(result.viral_score, result.score_breakdown)
        self._header.setText(f"{result.title}\nViral Score {result.viral_score} · {result.source_video} · {result.time_range_text}")
        titles = result.titles or ((result.title,) if result.title else ())
        for index, title in enumerate(titles, start=1):
            row = QHBoxLayout()
            label = QLabel(f"{index}. {title}")
            label.setWordWrap(True)
            label.setTextInteractionFlags(label.textInteractionFlags() | label.textInteractionFlags().TextSelectableByMouse)
            row.addWidget(label, stretch=1)
            row.addWidget(_copy_button(lambda t=title: t, "Копировать"))
            self._titles_box.addLayout(row)

        issue = result.llm_issue or (None if result.titles else LLMIssue(
            "unavailable", "Заголовки от LLM не получены.",
            "Запустите LM Studio (и проверьте, что модель помещается в память), затем обработайте видео снова.",
        ))
        self._llm_note.setText(f"⚠ {issue.message}\nЧто сделать: {issue.hint}" if issue else "")
        self._llm_note.setVisible(issue is not None)
        self._description.setPlainText(result.description)
        self._hashtags.setPlainText(result.hashtags_text)
        self._transcript.setText(result.transcript or "— речи в клипе нет —")

    def open_folder(self) -> None:
        if self._result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.folder)))
