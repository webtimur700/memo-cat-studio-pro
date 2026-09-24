import json
from pathlib import Path

from core.entities.clip import Clip
from core.entities.moment import Moment
from ui.viewmodels.clip_results import ClipResult, load_results_from_dir


def _write_clip(folder: Path, name: str, **overrides):
    (folder / f"{name}.mp4").write_bytes(b"x")
    (folder / f"{name}_cover.png").write_bytes(b"x")
    data = {
        "source_video": "v.mp4", "clip_file": f"{name}.mp4", "cover_file": f"{name}_cover.png",
        "start_sec": 65.0, "end_sec": 95.5, "viral_score": 77, "title": "T1",
        "titles": [f"Заголовок {i}" for i in range(1, 11)], "description": "Описание",
        "hashtags": ["#cat", "#funny"], "transcript": "привет",
    }
    data.update(overrides)
    (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_from_json_reads_everything(tmp_path):
    _write_clip(tmp_path, "a")
    result = ClipResult.from_json(tmp_path / "a.json")
    assert result.clip_path == tmp_path / "a.mp4" and result.cover_path == tmp_path / "a_cover.png"
    assert result.viral_score == 77 and len(result.titles) == 10
    assert result.hashtags_text == "#cat #funny"
    assert result.time_range_text == "01:05–01:35"
    assert result.folder == tmp_path


def test_missing_clip_file_is_skipped(tmp_path):
    _write_clip(tmp_path, "a")
    (tmp_path / "a.mp4").unlink()
    assert ClipResult.from_json(tmp_path / "a.json") is None


def test_corrupt_json_is_skipped_not_fatal(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    _write_clip(tmp_path, "ok")
    assert [r.clip_path.name for r in load_results_from_dir(tmp_path)] == ["ok.mp4"]


def test_missing_cover_is_none(tmp_path):
    _write_clip(tmp_path, "a")
    (tmp_path / "a_cover.png").unlink()
    assert ClipResult.from_json(tmp_path / "a.json").cover_path is None


def test_from_clip_and_full_text():
    clip = Clip(
        moment=Moment(1.0, 20.0, 80, 0.5, ()), output_path=Path("/x/c.mp4"), title="T",
        description="D", hashtags=("#a", "#b"), titles=("A", "B"),
    )
    result = ClipResult.from_clip(clip, "src.mp4")
    text = result.full_text()
    assert "1. A" in text and "2. B" in text and "D" in text and "#a #b" in text


def test_full_text_without_llm_uses_default_title():
    result = ClipResult(Path("/x/c.mp4"), "s", 0, 15, 50, "Момент 0s (score 50)")
    assert result.full_text() == "Заголовок: Момент 0s (score 50)"


def test_load_from_missing_dir_is_empty(tmp_path):
    assert load_results_from_dir(tmp_path / "nope") == []
