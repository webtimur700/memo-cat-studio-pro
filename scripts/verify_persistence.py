"""Проверка «настройки и история переживают перезапуск» настоящими процессами приложения (offscreen).

    python scripts/verify_persistence.py save DB OUT_DIR video.mp4   # 1-й запуск: меняет настройки, обрабатывает видео
    python scripts/verify_persistence.py show DB OUT_DIR             # 2-й запуск (новый процесс): что видно после перезапуска
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.main_window as mw  # noqa: E402


def main() -> None:
    mode, db, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    mw.DB_PATH, mw.EXPORT_OUTPUT_DIR = db, out
    app = QApplication([])
    window = mw.MainWindow()
    s = window.settings_view
    if mode == "save":
        s._fps_combo.setCurrentText("24")
        s._bitrate_spin.setValue(6)
        s._reserve_spin.setValue(4.5)
        s._music_offset_slider.setValue(-18)
        s._concurrent_spin.setValue(2)
        s._save_button.click()
        print("сохранено: fps=24, битрейт=6, запас=4.5 ГиБ, музыка −18 дБ, одновременно=2")
        window.project_view.videos_added.emit([Path(sys.argv[4]).resolve()])
        while not (window._scheduler.is_idle and not window._workers):
            QCoreApplication.processEvents()
            time.sleep(0.05)
        print("обработка завершена, проектов в истории:", len(window._history.projects()))
    else:
        e = window._settings.export
        print(f"после перезапуска: fps={e.fps} битрейт={e.bitrate_mbps} запас={window._settings.llm.pipeline_reserve_gb} ГиБ "
              f"музыка={window._settings.audio.music_offset_db} дБ одновременно={window._settings.batch.max_concurrent_videos}")
        print(f"поля в окне настроек: fps={s._fps_combo.currentText()} битрейт={s._bitrate_spin.value()} запас={s._reserve_spin.value()} "
              f"музыка={s._music_offset_slider.value()} одновременно={s._concurrent_spin.value()}")
        print("проекты:", window.project_view.project_texts())
        print("клипов в редакторе:", window.editor_view.clip_list.count())
        project = window._history.projects()[0]
        print("клипы проекта из базы:", [(c.title, c.viral_score, f"{c.start_sec:.0f}-{c.end_sec:.0f}s") for c in window._history.clips_for_project(project.id)])
        window._open_project(project.path)
        current = window.editor_view.current
        print("после двойного щелчка открыт клип:", current.clip_path.name if current else None, "| страница:", window._stack.currentIndex())
    window._db.close()


if __name__ == "__main__":
    main()
