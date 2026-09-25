"""Из чего сложился Viral Score момента: вклад каждого сигнала полосой и числом очков.

Пользователь по этому подбирает веса: видно, что именно (движение, животное в кадре, смены сцен, лай/мяуканье/смех,
прыжки и падения) дало очки и сколько. Сумма очков равна Viral Score (до округления).
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from core.entities.score import SignalPart

FORMULA_NOTE = "Оценка момента = 0.6 × лучшее окно + 0.4 × среднее по окнам момента."
WEIGHTS_HINT = "Влияние сигналов меняется в настройках: «Веса Viral Score»."


class ScoreBreakdownWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(4)
        self._grid.setColumnStretch(1, 1)
        self._rows: list[list[QWidget]] = []
        self._note = QLabel()
        self._note.setObjectName("clipMeta")
        self._note.setWordWrap(True)

    def _clear(self) -> None:
        for widgets in self._rows:
            for w in widgets:
                self._grid.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
        self._rows = []
        self._grid.removeWidget(self._note)

    def set_breakdown(self, score: int, parts: tuple[SignalPart, ...]) -> None:
        self._clear()
        if not parts:
            self._note.setText("Разложение оценки есть у клипов, обработанных новой версией; для этого клипа его нет.")
            self._grid.addWidget(self._note, 0, 0, 1, 3)
            return
        for row, part in enumerate(parts):
            name = QLabel(part.label + (" (бонус)" if part.bonus else ""))
            name.setObjectName("clipTitle")
            bar = QProgressBar()
            bar.setObjectName("scoreBar")
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setRange(0, 1000)
            bar.setValue(round(1000 * max(0.0, part.points) / max(1.0, float(score))))
            detail = f"{part.detail} · " if part.detail else ""
            points = QLabel(f"+{part.points:.1f}")
            points.setObjectName("clipScore")
            points.setToolTip(f"{detail}выражен на {part.value:.0%}, вес {part.weight:g}")
            info = QLabel(f"{detail}выражен на {part.value:.0%} · вес {part.weight:g}")
            info.setObjectName("clipMeta")
            self._grid.addWidget(name, row * 2, 0)
            self._grid.addWidget(bar, row * 2, 1)
            self._grid.addWidget(points, row * 2, 2)
            self._grid.addWidget(info, row * 2 + 1, 0, 1, 3)
            self._rows.append([name, bar, points, info])
        total = sum(p.points for p in parts)
        self._note.setText(f"Итого {total:.0f} из 100. {FORMULA_NOTE} {WEIGHTS_HINT}")
        self._grid.addWidget(self._note, len(parts) * 2, 0, 1, 3)

    def row_texts(self) -> list[str]:
        """Для тестов: «название|очки|подробности» по каждой строке."""
        return [f"{w[0].text()}|{w[2].text()}|{w[3].text()}" for w in self._rows]

    def note_text(self) -> str:
        return self._note.text()
