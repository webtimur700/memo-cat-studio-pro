"""Эффекты плагинов реально применяются к готовому клипу; без выбранных эффектов дополнительный проход не запускается."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import export.export_service as export_service
from core.entities.settings import ExportSettings, PluginSettings, UserSettings
from core.entities.subtitle import SubtitleSegment, WordTiming
from effects.branding_overlay import BrandingOverlay, logo_xy, resolve_logo_path
from effects.safe_zone import SafeZone
from export.dynamic_crop import CropSample
from export.export_service import ExportPlan, ExportService
from plugins.loader import load_plugins
from plugins.sdk.registry import PluginRegistry
from subtitles.ass_renderer import render_ass
from vision.smart_crop import CropWindow

EXAMPLES = Path(__file__).resolve().parents[2] / "plugins" / "examples"


@pytest.fixture(scope="module")
def colorful_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fx") / "colorful.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:duration=6:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
                    "-shortest", str(path)], check=True)
    return path


def _sepia():
    registry = PluginRegistry()
    load_plugins(EXAMPLES, registry)
    return registry.get_effect("vintage_sepia")


def _plan(video: Path, out: Path, tmp_path: Path, effects=(), subtitles=True) -> ExportPlan:
    samples = [CropSample(float(i), CropWindow(x1=100, y1=0, x2=505, y2=720, zoom_factor=1.0, timestamp_sec=float(i))) for i in range(4)]
    ass = None
    if subtitles:
        ass = tmp_path / "subs.ass"
        ass.write_text(render_ass([SubtitleSegment(words=[WordTiming("Привет", 0.2, 3.0)])]), encoding="utf-8")
    frame_size = (1080, 1920)
    zone = SafeZone.from_settings(frame_size, UserSettings().safe_zone)
    return ExportPlan(
        source_path=video, output_path=out, crop_samples=samples, subtitle_ass_path=ass, banner_rect=None, banner_text_lines=[],
        banner_appear_at_sec=4.0, banner_duration_sec=5.0, branding=BrandingOverlay.build(frame_size=frame_size, logo_path=resolve_logo_path(tmp_path / "assets"), logo_position="top_right", subscribe_enabled=False, zone=zone),
        settings=ExportSettings(width=1080, height=1920, quality_preset="low"), effects=tuple(effects),
    )


def _frame(clip: Path, t: float, tmp_path: Path) -> np.ndarray:
    png = tmp_path / f"f_{clip.stem}_{t}.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1", str(png)], check=True)
    return np.asarray(Image.open(png).convert("RGB")).astype(int)


def _sepia_share(frame: np.ndarray) -> float:
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    return float(((r >= g - 2) & (g >= b - 2)).mean())


def test_sepia_is_visible_in_the_finished_clip_and_branding_keeps_its_colours(colorful_video, tmp_path):
    plain = ExportService().export_clip(_plan(colorful_video, tmp_path / "plain.mp4", tmp_path))
    toned = ExportService().export_clip(_plan(colorful_video, tmp_path / "toned.mp4", tmp_path, effects=[("vintage_sepia", _sepia())]))

    a, b = _frame(plain, 1.0, tmp_path), _frame(toned, 1.0, tmp_path)
    body = (slice(700, 1500), slice(100, 950))            # видеоряд без зоны логотипа и субтитров
    assert _sepia_share(b[body]) > 0.97 and _sepia_share(a[body]) < 0.6         # у сепии всегда R >= G >= B, у testsrc2 нет
    assert (b[body][..., 0] - b[body][..., 2]).mean() > 25                       # тёплый оттенок заметен глазом
    assert np.abs(a[body] - b[body]).mean() > 15                                 # кадр реально отличается

    # логотип (фиолетовый) сепией не тонируется: эффект применён до оверлеев
    zone = SafeZone.from_settings((1080, 1920), UserSettings().safe_zone)
    logo_w, logo_h = int(1080 * 0.19), int(1080 * 0.19 * 240 / 640)
    x, y = logo_xy("top_right", zone, (logo_w, logo_h))
    r, g, bl = b[y + logo_h - 10, x + 8]
    assert (abs(r - 124) < 35, abs(g - 92) < 35, abs(bl - 255) < 35) == (True, True, True)

    # звук на месте
    audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(toned)],
                           capture_output=True, text=True).stdout.strip()
    assert audio == "aac"


def test_without_effects_there_is_no_extra_pass_and_with_effects_subtitles_move_after_the_effect(colorful_video, tmp_path, monkeypatch):
    commands = []
    real_run = export_service._run
    monkeypatch.setattr(export_service, "_run", lambda cmd, cwd=None: (commands.append(" ".join(cmd)), real_run(cmd, cwd))[1])

    def forbidden(*args, **kwargs):
        raise AssertionError("проход с эффектами не должен запускаться без выбранных эффектов")

    monkeypatch.setattr(export_service, "apply_effects_to_video", forbidden)
    ExportService().export_clip(_plan(colorful_video, tmp_path / "a.mp4", tmp_path))
    assert len(commands) == 3 and "ass=subs.ass" in commands[0]                  # как раньше: база с субтитрами, ролик оверлеев, склейка

    monkeypatch.undo()
    commands.clear()
    monkeypatch.setattr(export_service, "_run", lambda cmd, cwd=None: (commands.append(" ".join(cmd)), real_run(cmd, cwd))[1])
    ExportService().export_clip(_plan(colorful_video, tmp_path / "b.mp4", tmp_path, effects=[("vintage_sepia", _sepia())]))
    assert "ass=" not in commands[0] and "ass=subs.ass" in commands[-1]           # текст не тонируется, вжигается после эффекта


def test_effects_run_in_the_selected_order_and_a_broken_effect_is_skipped(colorful_video, tmp_path):
    calls = []

    def first(frame):
        calls.append("first")
        return frame

    def broken(frame):
        calls.append("broken")
        raise RuntimeError("плагин сломан")

    def wrong_shape(frame):
        calls.append("wrong")
        return frame[:10]

    def last(frame):
        calls.append("last")
        return 255 - frame

    out = ExportService().export_clip(_plan(colorful_video, tmp_path / "o.mp4", tmp_path, subtitles=False,
                                            effects=[("first", first), ("broken", broken), ("wrong", wrong_shape), ("last", last)]))
    assert out.exists()
    assert calls[:4] == ["first", "broken", "wrong", "last"]           # порядок из настроек; сломанные отключаются после первого кадра
    assert "broken" not in calls[4:] and "wrong" not in calls[4:] and calls[4:6] == ["first", "last"]
    plain = ExportService().export_clip(_plan(colorful_video, tmp_path / "p.mp4", tmp_path, subtitles=False))
    body = (slice(700, 1500), slice(100, 950))
    assert np.abs(_frame(out, 1.0, tmp_path)[body] - (255 - _frame(plain, 1.0, tmp_path)[body])).mean() < 25   # «last» (негатив) применился


def test_pipeline_applies_selected_plugin_effect_and_records_it(colorful_video, tmp_path, monkeypatch):
    from dataclasses import replace

    from core.entities.settings import AudioSettings, ShortsSettings, ViralScoreSettings
    from pipeline.pipeline_runner import PipelineRunner

    import subtitles.subtitle_service as subtitle_service

    class _Silent:
        def __init__(self, **_):
            pass

        def transcribe(self, audio_path, language=None):
            return []

    monkeypatch.setattr(subtitle_service, "WhisperTranscriber", _Silent)
    registry = PluginRegistry()
    load_plugins(EXAMPLES, registry)
    settings = replace(
        UserSettings(), shorts=ShortsSettings(allowed_durations_sec=(5,), min_duration_sec=5, max_duration_sec=5),
        viral_score=ViralScoreSettings(queue_threshold=0), audio=AudioSettings(music_enabled=False),
        plugins=PluginSettings(enabled_effects=("vintage_sepia", "no_such_plugin")),
    )
    clips = PipelineRunner(models_dir=tmp_path / "none", output_dir=tmp_path / "out", plugin_registry=registry).process_video(colorful_video, settings)
    assert clips
    meta = json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))
    assert meta["effects"] == ["vintage_sepia"]                        # несуществующий плагин пропущен, клип собран
    assert _sepia_share(_frame(clips[0].output_path, 1.0, tmp_path)[700:1500, 100:950]) > 0.97
