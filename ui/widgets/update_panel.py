"""Кнопка «Проверить обновления» (пункт ТЗ «АВТООБНОВЛЕНИЕ»): только по нажатию, git работает в фоновом потоке.

Проверка (git fetch) показывает, есть ли новые коммиты; «Обновить» доступна, только если обновление безопасно
(нет незакоммиченных изменений, ветка не разошлась), и спрашивает подтверждение со списком новых коммитов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from updater.git_updater import UpdateResult, UpdateStatus, apply_update, check_for_updates


class _GitJob(QThread):
    done = Signal(object)

    def __init__(self, function: Callable[[Path], object], repo: Path, parent=None) -> None:
        super().__init__(parent)
        self._function, self._repo = function, repo

    def run(self) -> None:
        try:
            self.done.emit(self._function(self._repo))
        except Exception as exc:   # git упал неожиданным образом — показываем причину, а не молчим
            self.done.emit(UpdateStatus(error=str(exc)))


class UpdatePanel(QWidget):
    def __init__(self, repo: Path, parent: QWidget | None = None, checker=check_for_updates, applier=apply_update) -> None:
        super().__init__(parent)
        self._repo, self._checker, self._applier = repo, checker, applier
        self._job: _GitJob | None = None
        self._status: UpdateStatus | None = None
        self.busy_check: Callable[[], bool] = lambda: False           # MainWindow: идёт ли очередь
        self.confirm: Callable[[UpdateStatus], bool] = self._ask_user  # тесты подменяют

        self._check_button = QPushButton("Проверить обновления")
        self._check_button.clicked.connect(self.check_now)
        self._update_button = QPushButton("Обновить")
        self._update_button.setEnabled(False)
        self._update_button.clicked.connect(self.update_now)
        self._label = QLabel("Проверка запускается только по кнопке.")
        self._label.setWordWrap(True)
        self._label.setObjectName("clipMeta")

        buttons = QHBoxLayout()
        buttons.addWidget(self._check_button)
        buttons.addWidget(self._update_button)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self._label)

    # --- действия
    def check_now(self) -> None:
        self._start(self._checker, self._on_checked, "Проверяю обновления (git fetch)…")

    def update_now(self) -> None:
        status = self._status
        if status is None or not status.can_update:
            return
        if self.busy_check():
            self._label.setText("Идёт обработка видео — дождитесь конца очереди и обновите после этого.")
            return
        if not self.confirm(status):
            self._label.setText("Обновление отменено.")
            return
        self._start(self._applier, self._on_updated, "Обновляю…")

    def _start(self, function, slot, text: str) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self._label.setText(text)
        self._check_button.setEnabled(False)
        self._update_button.setEnabled(False)
        self._job = _GitJob(function, self._repo, self)
        self._job.done.connect(slot)
        self._job.start()

    def wait(self, timeout_ms: int = 60000) -> None:
        """Для тестов: дождаться завершения фонового git-задания."""
        if self._job is not None:
            self._job.wait(timeout_ms)

    # --- результаты (UI-поток)
    def _on_checked(self, status: UpdateStatus) -> None:
        self._status = status
        text = status.message
        if status.can_update and status.new_commits:
            text += "\n" + "\n".join(f"• {c}" for c in status.new_commits[:8]) + ("\n…" if status.behind > 8 else "")
        self._label.setText(text)
        self._check_button.setEnabled(True)
        self._update_button.setEnabled(status.can_update)

    def _on_updated(self, result) -> None:
        self._check_button.setEnabled(True)
        self._update_button.setEnabled(False)
        self._status = None
        self._label.setText(result.message if isinstance(result, UpdateResult) else result.message)

    def _ask_user(self, status: UpdateStatus) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Обновить приложение?")
        box.setText(f"Будет загружено {status.behind} нов. коммит(ов) из {status.upstream}. Локальные файлы не пострадают: "
                    f"обновление только «вперёд» (fast-forward). После обновления приложение нужно перезапустить.")
        box.setDetailedText("\n".join(status.new_commits))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    # --- для тестов
    def label_text(self) -> str:
        return self._label.text()

    def buttons_enabled(self) -> tuple[bool, bool]:
        return self._check_button.isEnabled(), self._update_button.isEnabled()
