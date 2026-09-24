import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio.event_classifier import (
    FRAME_HOP_SEC, FRAME_SAMPLES, HOP_SAMPLES, AudioEventClassifier, AudioEventTimeline,
)
from scoring.viral_score_service import ScoreInputs, compute_viral_score
from core.entities.settings import ViralScoreSettings

CLASS_MAP = "index,mid,display_name\n0,/m/09x0r,Speech\n1,/m/05tny_,Bark\n2,/m/07qrkrw,Meow\n3,/m/01j3sz,Laughter\n4,/m/0,Silence\n"


class _FakeSession:
    """Вместо YAMNet: число окон как у настоящей модели, вероятности классов задаёт тест по номеру окна."""

    class _Input:
        name = "waveform"

    def __init__(self, hot: dict[int, tuple[int, float]] | None = None):
        self.hot = hot or {}
        self.offset = 0

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feed):
        n = (feed["waveform"].size - FRAME_SAMPLES) // HOP_SAMPLES + 1
        probs = np.zeros((n, 5), dtype=np.float32)
        probs[:, 0] = 0.9   # речь везде — на оценку событий влиять не должна
        for frame, (cls, p) in self.hot.items():
            if self.offset <= frame < self.offset + n:
                probs[frame - self.offset, cls] = p
        self.offset += 125 if n >= 125 else n
        return [probs]


@pytest.fixture()
def classifier(tmp_path):
    (tmp_path / "yamnet_class_map.csv").write_text(CLASS_MAP, encoding="utf-8")
    def make(hot=None):
        return AudioEventClassifier(tmp_path, session=_FakeSession(hot))
    return make


def _video(path: Path, seconds: float, audio: bool = True) -> Path:
    args = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=64x64:duration={seconds}:rate=5"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    subprocess.run(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_no_frames_lost_or_duplicated_across_60s_chunks(classifier, tmp_path):
    video = _video(tmp_path / "long.mp4", 130.0)
    timeline = classifier().analyze_video(video)
    samples = int(130.0 * 16000)
    assert abs(timeline.frame_scores.size - ((samples - FRAME_SAMPLES) // HOP_SAMPLES + 1)) <= 3   # допуск на округление ffmpeg


def test_event_position_is_reported_at_the_right_time_even_after_a_chunk_boundary(classifier, tmp_path):
    video = _video(tmp_path / "long.mp4", 130.0)
    meow_frame = 190                       # 190 * 0.48 = 91.2 с — во втором минутном куске
    timeline = classifier({meow_frame: (2, 0.6)}).analyze_video(video)
    t = meow_frame * FRAME_HOP_SEC + FRAME_HOP_SEC
    assert timeline.score_between(t - 1, t + 1) == pytest.approx(1.0 / 3, abs=0.02)   # одно окно из трёх сильнейших
    assert timeline.score_between(10, 20) == 0.0
    assert timeline.dominant_class_between(t - 1, t + 1) == "Meow"


def test_speech_alone_is_not_an_event_and_saturation_is_capped(classifier, tmp_path):
    video = _video(tmp_path / "v.mp4", 12.0)
    quiet = classifier().analyze_video(video)
    assert quiet.frame_scores.max() == 0.0                       # «Speech» 0.9 не считается событием
    loud = classifier({4: (1, 0.9), 5: (1, 0.9), 6: (1, 0.9)}).analyze_video(video)
    assert loud.score_between(2.0, 4.0) == pytest.approx(1.0)    # три сильных «Bark» подряд — максимум, не больше


def test_video_without_audio_gives_empty_timeline_and_zero_scores(classifier, tmp_path):
    timeline = classifier().analyze_video(_video(tmp_path / "mute.mp4", 5.0, audio=False))
    assert timeline.frame_scores.size == 0 and timeline.score_between(0, 5) == 0.0
    assert AudioEventTimeline(np.zeros(0, dtype=np.float32), ()).dominant_class_between(0, 5) == ""


def test_missing_model_degrades_to_none(tmp_path):
    assert AudioEventClassifier.load(tmp_path / "no_yamnet") is None


def test_viral_score_uses_audio_only_when_available():
    weights = ViralScoreSettings()
    silent = ScoreInputs(motion_intensity=0.4, detection_presence=0.5, scene_change_count=0)
    same_without = compute_viral_score(silent, weights)
    with_zero_audio = compute_viral_score(ScoreInputs(0.4, 0.5, 0, audio_event=0.0), weights)
    with_event = compute_viral_score(ScoreInputs(0.4, 0.5, 0, audio_event=1.0), weights)
    assert with_event > same_without > with_zero_audio           # лай поднимает оценку; тишина при рабочем классификаторе — опускает
    assert compute_viral_score(silent, weights) == same_without  # без классификатора (None) формула как раньше


YAMNET_DIR = Path(__file__).resolve().parents[2] / "models" / "yamnet"


@pytest.mark.skipif(not (YAMNET_DIR / "yamnet.onnx").exists(), reason="нет models/yamnet (scripts/download_models.py)")
def test_real_yamnet_runs_on_cpu_and_tone_is_not_an_animal_event(tmp_path):
    import time

    real = AudioEventClassifier.load(YAMNET_DIR)
    assert real is not None
    video = _video(tmp_path / "tone.mp4", 30.0)
    t0 = time.perf_counter()
    timeline = real.analyze_video(video)
    elapsed = time.perf_counter() - t0
    assert timeline.frame_scores.size > 55
    assert elapsed < 10.0                                    # 30 с звука — доли секунды на CPU
    assert timeline.frame_scores.max() < 0.5                 # чистый тон — не лай и не мяуканье
