import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from ui.viewmodels.clip_results import ClipResult, load_results_from_dir
from ui.views.editor_view import EditorView

TITLES = [f"Заголовок номер {i}" for i in range(1, 11)]


@pytest.fixture()
def output_dir(tmp_path) -> Path:
    for index, (start, score) in enumerate([(5.0, 72), (40.0, 88)], start=1):
        name = f"vid_moment{index}"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=270x480:duration=1:rate=10",
             "-pix_fmt", "yuv420p", str(tmp_path / f"{name}.mp4")], check=True,
        )
        Image.new("RGB", (540, 960), (30 * index, 90, 160)).save(tmp_path / f"{name}_cover.png")
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "source_video": "vid.mp4", "clip_file": f"{name}.mp4", "cover_file": f"{name}_cover.png",
            "start_sec": start, "end_sec": start + 15, "viral_score": score, "title": TITLES[0],
            "titles": TITLES, "description": "SEO описание про кота", "hashtags": ["#кот", "#cat", "#funny"],
            "transcript": "привет мир",
        }, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _buttons(widget, text_start: str) -> list[QPushButton]:
    return [b for b in widget.findChildren(QPushButton) if b.text().startswith(text_start)]


def test_results_listed_with_score_and_cover(qapp, output_dir):
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    assert view.clip_list.count() == 2
    texts = " ".join(label.text() for label in view.clip_list.findChildren(QLabel))
    assert "Viral Score 72" in texts and "Viral Score 88" in texts
    thumbs = [l for l in view.clip_list.findChildren(QLabel) if l.objectName() == "clipThumb"]
    assert len(thumbs) == 2 and all(t.pixmap() is not None and not t.pixmap().isNull() for t in thumbs)


def test_selection_loads_clip_into_player_and_shows_details(qapp, output_dir):
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    view.clip_list.setCurrentRow(1)
    current = view.current
    assert view.preview.source_path == current.clip_path
    header = view.details._header.text()
    assert current.title in header and f"Viral Score {current.viral_score}" in header
    # ровно 10 заголовков, у каждого своя кнопка копирования
    title_labels = [l for l in view.details.findChildren(QLabel) if l.text().split(".")[0].isdigit()]
    assert len(title_labels) == 10
    assert len(_buttons(view.details, "Копировать")) >= 10 + 2   # 10 заголовков + описание + хештеги


def test_copy_buttons_put_text_on_clipboard(qapp, output_dir):
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    clipboard = QApplication.clipboard()

    title_buttons = view.details._titles_container.findChildren(QPushButton)
    assert len(title_buttons) == 10
    title_buttons[0].click()
    assert clipboard.text() == TITLES[0]
    title_buttons[2].click()
    assert clipboard.text() == TITLES[2]

    view.details._description_copy.click()
    assert clipboard.text() == "SEO описание про кота"
    view.details._hashtags_copy.click()
    assert clipboard.text() == "#кот #cat #funny"
    view.details._copy_all_button.click()
    text = clipboard.text()
    assert "10. Заголовок номер 10" in text and "SEO описание" in text and "#кот #cat #funny" in text


def test_open_folder_button_opens_output_dir(qapp, output_dir, monkeypatch):
    opened = []
    import ui.widgets.clip_details as details_module

    monkeypatch.setattr(details_module.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    view.details._open_folder_button.click()
    assert opened == [str(output_dir)]


def test_clip_without_llm_shows_note_and_default_title(qapp, tmp_path):
    result = ClipResult(tmp_path / "c.mp4", "v.mp4", 0.0, 15.0, 50, "Момент 0s (score 50)")
    view = EditorView()
    view.details.set_clip(result)
    assert "Заголовки от LLM не получены" in view.details._llm_note.text()
    assert len(_buttons(view.details, "Копировать")) >= 1


def test_no_selection_state_is_disabled(qapp):
    view = EditorView()
    assert not view.details._open_folder_button.isEnabled()
    assert not view.details._copy_all_button.isEnabled()


def test_timeline_click_selects_matching_clip(qapp, output_dir):
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    view.clip_list.setCurrentRow(0)
    view._on_timeline_moment_selected(40.0, 55.0)
    assert view.current.start_sec == 40.0


def test_duplicates_are_not_added_twice(qapp, output_dir):
    view = EditorView()
    results = load_results_from_dir(output_dir)
    view.add_results(results)
    view.add_results(results)
    assert view.clip_list.count() == 2


def test_timeline_fits_all_moments_in_view_without_scrolling(qapp):
    from ui.views.timeline_view import TimelineMoment, TimelineView

    view = TimelineView()
    view.resize(900, 100)
    view.show()
    qapp.processEvents()
    view.set_moments([TimelineMoment(101.0, 120.0, 79, "a"), TimelineMoment(150.0, 165.0, 83, "b")], 165.0)
    viewport_width = view._view.viewport().width()
    assert view._scene.sceneRect().width() <= viewport_width + 5          # всё видно без прокрутки
    assert not view._view.horizontalScrollBar().isVisible() or view._view.horizontalScrollBar().maximum() == 0
    # последний момент целиком внутри видимой области
    last = view._items[-1].sceneBoundingRect()
    assert last.right() <= viewport_width + 5


def test_timeline_refits_on_resize_but_not_after_user_zoom(qapp):
    from ui.views.timeline_view import TimelineMoment, TimelineView

    view = TimelineView()
    view.resize(600, 100)
    view.show()
    view.set_moments([TimelineMoment(10.0, 30.0, 70, "a")], 100.0)
    narrow = view._pixels_per_second
    view.resize(1200, 100)
    qapp.processEvents()
    assert view._pixels_per_second > narrow                                # подстроился под ширину
    view._auto_fit = False
    fixed = view._pixels_per_second
    view.resize(700, 100)
    qapp.processEvents()
    assert view._pixels_per_second == fixed                                # ручной масштаб не трогаем


def test_selecting_clip_highlights_its_moment_on_timeline(qapp, output_dir):
    view = EditorView()
    view.add_results(load_results_from_dir(output_dir))
    view.clip_list.setCurrentRow(1)
    selected = [i.moment.start_sec for i in view.timeline._scene.selectedItems() if hasattr(i, "moment")]
    assert selected == [view.current.start_sec]


def test_selection_signal_is_emitted_once_per_click(qapp):
    from ui.views.timeline_view import TimelineMoment, TimelineView

    view = TimelineView()
    emitted = []
    view.moment_selected.connect(lambda s, e: emitted.append(s))
    for _ in range(3):   # раньше каждое set_moments добавляло ещё одно подключение сигнала
        view.set_moments([TimelineMoment(10.0, 30.0, 70, "a"), TimelineMoment(40.0, 60.0, 80, "b")], 100.0)
    view._items[1].setSelected(True)
    assert emitted == [40.0]


def test_switching_clip_after_playback_started_does_not_freeze_the_ui(tmp_path):
    """Смена клипа при открытом другом раньше вешала интерфейс (GIL-дедлок Qt Multimedia на реальном
    h264 1080x1920). Прогон в подпроцессе с таймаутом: зависание = провал, а не вечный тест."""
    import subprocess
    import sys
    import textwrap

    for name in ("a", "b"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:duration=3:rate=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-c:a", "aac",
             "-pix_fmt", "yuv420p", str(tmp_path / f"{name}.mp4")], check=True,
        )
    script = textwrap.dedent(f"""
        import os, time
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from pathlib import Path
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        from ui.views.preview_player import PreviewPlayer
        app = QApplication([])
        player = PreviewPlayer()
        def spin(sec):
            end = time.time() + sec
            while time.time() < end:
                QCoreApplication.processEvents(); time.sleep(0.01)
        for name in ("a", "b", "a", "b"):
            player.load_video(Path(r"{tmp_path}") / (name + ".mp4"))
            spin(1.5)
        player.load_video(Path(r"{tmp_path}") / "a.mp4", autoplay=True)
        spin(0.5)
        print("OK")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=90,
                            cwd=Path(__file__).resolve().parents[2])
    assert "OK" in result.stdout, result.stderr[-500:]


def test_settings_view_saves_music_and_queue_options(qapp):
    from core.entities.settings import UserSettings
    from ui.views.settings_view import SettingsView

    view = SettingsView(UserSettings())
    received = []
    view.settings_saved.connect(received.append)
    view._music_checkbox.setChecked(False)
    view._music_offset_slider.setValue(-18)
    view._duck_checkbox.setChecked(False)
    view._concurrent_spin.setValue(3)
    view._save_button.click()
    saved = received[0]
    assert (saved.audio.music_enabled, saved.audio.music_offset_db, saved.audio.duck_on_speech) == (False, -18.0, False)
    assert saved.batch.max_concurrent_videos == 3


def test_settings_view_saves_llm_reserve(qapp):
    from core.entities.settings import UserSettings
    from ui.views.settings_view import SettingsView

    view = SettingsView(UserSettings())
    assert view._reserve_spin.value() == 6.0
    got = []
    view.settings_saved.connect(got.append)
    view._reserve_spin.setValue(4.5)
    view._save_button.click()
    assert got[0].llm.pipeline_reserve_gb == 4.5
