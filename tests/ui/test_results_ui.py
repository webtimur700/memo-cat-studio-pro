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
    assert "LM Studio была недоступна" in view.details._llm_note.text()
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
