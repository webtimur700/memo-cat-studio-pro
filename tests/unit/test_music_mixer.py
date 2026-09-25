import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio.music_mixer import (
    build_duck_expression, list_tracks, mix_music, pick_track, speech_intervals,
)
from core.entities.subtitle import WordTiming


def test_speech_intervals_merge_close_words_and_pad():
    words = [WordTiming("а", 1.0, 1.3), WordTiming("б", 1.5, 2.0), WordTiming("в", 5.0, 5.5)]
    intervals = speech_intervals(words, clip_duration_sec=6.0)
    assert intervals == [pytest.approx((0.85, 2.15)), pytest.approx((4.85, 5.65))]
    assert speech_intervals([], 6.0) == []
    assert speech_intervals([WordTiming("а", 0.0, 6.0)], 6.0) == [(0.0, 6.0)]   # края клипа не выходят за 0..длительность


def test_duck_expression_empty_without_speech_and_gain_matches_db():
    assert build_duck_expression([], -12) is None
    expr = build_duck_expression([(1.0, 2.0)], -12)
    assert expr.startswith("1-0.7488*")   # 1 - 10^(-12/20)
    assert "max(" in build_duck_expression([(1.0, 2.0), (4.0, 5.0)], -12)


def test_list_tracks_ignores_non_audio_and_missing_folder(tmp_path):
    (tmp_path / "b.mp3").write_bytes(b"x")
    (tmp_path / "a.WAV").write_bytes(b"x")
    (tmp_path / "README.md").write_text("x")
    assert [p.name for p in list_tracks(tmp_path)] == ["a.WAV", "b.mp3"]
    assert list_tracks(tmp_path / "nope") == []


def test_pick_track_is_stable_prefers_long_enough_and_handles_empty(tmp_path):
    short, long_ = tmp_path / "short.mp3", tmp_path / "long.mp3"
    durations = {short: 20.0, long_: 90.0}
    picks = {pick_track([short, long_], f"clip{i}", 45.0, durations) for i in range(20)}
    assert picks == {long_}                                     # короткий не берём, пока есть подходящий
    assert pick_track([short, long_], "x", 45.0, durations) == pick_track([short, long_], "x", 45.0, durations)
    assert pick_track([short], "x", 45.0, {short: 20.0}) == short   # других нет — зациклится
    assert pick_track([], "x", 45.0) is None


SINE_AMPLITUDE = 0.125   # lavfi sine по умолчанию 1/8 от полной шкалы


def _run(*args):
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", *args], check=True)


def _pcm(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _tone_level(signal: np.ndarray, start: float, end: float, freq: float) -> float:
    """Амплитуда синусоиды freq в отрезке (проекция на sin/cos) — не путает музыку с оригиналом."""
    seg = signal[int(start * 16000): int(end * 16000)]
    t = np.arange(seg.size) / 16000
    return float(2 * np.hypot((seg * np.sin(2 * np.pi * freq * t)).mean(), (seg * np.cos(2 * np.pi * freq * t)).mean()))


@pytest.fixture()
def clip_and_track(tmp_path):
    clip, track = tmp_path / "clip.mp4", tmp_path / "track.wav"
    _run("-f", "lavfi", "-i", "testsrc2=size=320x240:duration=8:rate=10", "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest", str(clip))
    _run("-f", "lavfi", "-i", "sine=frequency=1000:duration=3", str(track))     # короче клипа: проверяем зацикливание
    return clip, track


def test_plan_gain_puts_music_offset_below_the_clip_regardless_of_track_mastering():
    from audio.music_mixer import MUSIC_ONLY_LUFS, plan_gain

    gain, target = plan_gain(clip_lufs=-29.0, track_loudness=-12.0, offset_db=-14)
    assert target == pytest.approx(-43.0) and gain == pytest.approx(-31.0)
    gain, target = plan_gain(clip_lufs=-16.0, track_loudness=-12.0, offset_db=-14)   # громкий клип — музыка громче
    assert target == pytest.approx(-30.0) and gain == pytest.approx(-18.0)
    assert plan_gain(-16.0, -30.0, -14)[0] == pytest.approx(0.0 + (-30.0 - -30.0))   # тихий трек: подтянуть ровно до цели
    assert plan_gain(None, -12.0, -14)[1] == MUSIC_ONLY_LUFS                          # у клипа нет звука — музыка сама по себе
    assert plan_gain(-70.0 + 5, -12.0, -14)[1] == MUSIC_ONLY_LUFS                     # почти тишина — тоже
    assert plan_gain(-20.0, -60.0, -3)[0] == 15.0                                     # усиление ограничено


def test_measure_lufs_reads_tone_level_and_none_for_silence_and_no_audio(tmp_path):
    from audio.music_mixer import measure_lufs

    loud, quiet, silent = tmp_path / "l.wav", tmp_path / "q.wav", tmp_path / "s.wav"
    _run("-f", "lavfi", "-i", "sine=frequency=1000:duration=3", str(loud))
    _run("-f", "lavfi", "-i", "sine=frequency=1000:duration=3", "-af", "volume=-20dB", str(quiet))
    _run("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(silent))
    assert measure_lufs(loud) - measure_lufs(quiet) == pytest.approx(20.0, abs=0.3)
    assert measure_lufs(silent) is None
    video = tmp_path / "v.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=64x64:duration=2:rate=5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video))
    assert measure_lufs(video) is None


def _db(a, b):
    return 20 * np.log10(a / b)


def test_music_sits_offset_below_the_original_and_is_ducked_only_under_speech(clip_and_track):
    clip, track = clip_and_track
    video_before = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name,nb_frames", "-of", "csv=p=0", str(clip)],
                                  capture_output=True, text=True).stdout
    # речь на 3.0-5.0 с (с полями приглушения ~2.85-5.15)
    report = mix_music(clip, track, clip_duration_sec=8.0, offset_db=-12, duck_intervals=[(2.85, 5.15)], duck_db=-12)
    signal = _pcm(clip)

    original = _tone_level(signal, 1.0, 7.0, 300)
    free = _tone_level(signal, 1.0, 2.5, 1000)     # музыка вне речи
    ducked = _tone_level(signal, 3.5, 4.5, 1000)   # музыка под речью
    later = _tone_level(signal, 6.0, 7.0, 1000)    # вернулась (трек короче клипа — зациклен)
    assert original == pytest.approx(SINE_AMPLITUDE, rel=0.2)                 # оригинальный звук на месте, не приглушён
    assert _db(free, original) == pytest.approx(-12, abs=1.5)                 # музыка на 12 дБ ниже оригинала (K-взвешивание ±)
    assert _db(ducked, free) == pytest.approx(-12, abs=1.5)                   # на речи ещё на 12 дБ тише
    assert later / free == pytest.approx(1.0, rel=0.15)
    assert report.offset_db == -12 and report.target_lufs == pytest.approx(report.clip_lufs - 12)

    video_after = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name,nb_frames", "-of", "csv=p=0", str(clip)],
                                 capture_output=True, text=True).stdout
    assert video_after == video_before               # видео скопировано без перекодирования
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                                    capture_output=True, text=True).stdout)
    assert duration == pytest.approx(8.0, abs=0.3)


def test_quiet_and_loud_clips_get_proportionally_quiet_and_loud_music(tmp_path, clip_and_track):
    _, track = clip_and_track
    levels = {}
    for name, volume_db in (("loud", 0), ("quiet", -20)):
        clip = tmp_path / f"{name}.mp4"
        _run("-f", "lavfi", "-i", "testsrc2=size=64x64:duration=6:rate=5", "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
             "-af", f"volume={volume_db}dB", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest", str(clip))
        mix_music(clip, track, clip_duration_sec=6.0, offset_db=-14, duck_intervals=[], duck_db=-12)
        signal = _pcm(clip)
        levels[name] = (_tone_level(signal, 1.5, 4.5, 1000), _tone_level(signal, 1.5, 4.5, 300))
    # музыка следует за клипом: у тихого клипа она тише на те же 20 дБ, а расстояние до оригинала одинаково
    assert _db(levels["loud"][0], levels["quiet"][0]) == pytest.approx(20, abs=1.5)
    assert _db(levels["loud"][0], levels["loud"][1]) == pytest.approx(-14, abs=1.5)
    assert _db(levels["quiet"][0], levels["quiet"][1]) == pytest.approx(-14, abs=1.5)


def test_music_fades_out_at_the_end_and_clip_without_audio_gets_music_only(tmp_path, clip_and_track):
    from audio.music_mixer import MUSIC_ONLY_LUFS, measure_lufs

    _, track = clip_and_track
    silent = tmp_path / "silent.mp4"
    _run("-f", "lavfi", "-i", "testsrc2=size=320x240:duration=4:rate=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent))
    report = mix_music(silent, track, clip_duration_sec=4.0, offset_db=-14, duck_intervals=[], duck_db=-12)
    assert report.clip_lufs is None and report.target_lufs == MUSIC_ONLY_LUFS
    signal = _pcm(silent)
    assert measure_lufs(silent, 0.5, 2.5) == pytest.approx(MUSIC_ONLY_LUFS, abs=1.0)
    assert _tone_level(signal, 3.8, 4.0, 1000) < 0.3 * _tone_level(signal, 1.0, 2.0, 1000)   # затухание в конце
