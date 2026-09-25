import pytest
from dataclasses import replace

from core.entities.settings import ShortsSettings, UserSettings, ViralScoreSettings
from cutting.clip_selector_service import WindowScore, select_moments


def _windows(scores: list[int], step: float = 5.0) -> list[WindowScore]:
    return [WindowScore(i * step, (i + 1) * step, s, 0.5) for i, s in enumerate(scores)]


def _settings(**viral) -> UserSettings:
    base = UserSettings()
    return replace(base, viral_score=replace(base.viral_score, **viral)) if viral else base


def test_flat_good_video_is_not_cut_into_everything():
    # все окна выше абсолютного порога — раньше это давало клипы на 75% видео
    windows = _windows([60 + (i * 7) % 15 for i in range(36)])   # 180 с, score 60..74
    moments = select_moments(windows, _settings(), 180.0)
    total = sum(m.duration_sec for m in moments)
    assert 0 < total <= 0.35 * 180 + 60          # потолок покрытия (первый клип может его превысить)
    assert len(moments) < 9


def test_relative_choice_picks_the_best_windows():
    scores = [56] * 12
    scores[3] = 95
    scores[9] = 90
    moments = select_moments(_windows(scores), _settings(), 60.0)
    assert any(m.start_sec <= 15 < m.end_sec for m in moments)   # окно с 95 попало
    assert all(m.viral_score >= 55 for m in moments)


def test_absolute_minimum_still_applies():
    moments = select_moments(_windows([30, 35, 40, 32, 38, 36]), _settings(queue_threshold=55), 30.0)
    assert moments == []


def test_adjacent_strong_windows_are_merged_into_one_moment():
    scores = [40] * 4 + [90, 92, 88, 91, 89] + [40] * 3     # 25 с сильного подряд
    moments = select_moments(_windows(scores), _settings(), 60.0)
    assert len(moments) == 1
    m = moments[0]
    assert m.start_sec <= 20 and m.end_sec >= 45             # захвачены все сильные окна
    assert 15 <= m.duration_sec <= 60


def test_short_peak_is_extended_to_min_duration_around_neighbours():
    scores = [30] * 5 + [95] + [70] * 3 + [30] * 3
    moments = select_moments(_windows(scores), _settings(), 60.0)
    m = moments[0]
    assert m.duration_sec >= 15 and m.start_sec <= 25 < m.end_sec


def test_overlong_run_is_trimmed_to_max_duration():
    scores = [90] * 20                                       # 100 с сильного подряд
    moments = select_moments(_windows(scores), _settings(), 100.0)
    assert moments and all(m.duration_sec <= 60 + 1e-6 for m in moments)


def test_moments_do_not_overlap_and_respect_max_count():
    scores = ([90] * 3 + [30] * 3) * 10
    moments = select_moments(_windows(scores), _settings(), 180.0, max_moments=3)
    assert len(moments) <= 3
    for a, b in zip(moments, moments[1:]):
        assert a.end_sec <= b.start_sec


def test_boundaries_snap_to_scene_cut():
    scores = [40] * 3 + [90] * 4 + [40] * 5
    moments = select_moments(_windows(scores), _settings(), 60.0, scene_boundaries=[13.0, 37.0])
    m = moments[0]
    assert m.start_sec == 13.0 or m.end_sec == 37.0 or (13.0 in (m.start_sec, m.end_sec))


def test_boundaries_snap_to_speech_pause_not_mid_phrase():
    scores = [40] * 3 + [90] * 4 + [40] * 5
    pauses = [14.2, 36.6]
    moments = select_moments(
        _windows(scores), _settings(), 60.0, pause_finder=lambda lo, hi: [p for p in pauses if lo <= p <= hi]
    )
    m = moments[0]
    assert m.start_sec in pauses or m.end_sec in pauses


def test_snapping_never_breaks_duration_limits():
    scores = [40] * 3 + [90] * 4 + [40] * 5
    moments = select_moments(_windows(scores), _settings(), 60.0, scene_boundaries=[15.5, 16.0, 17.0, 29.0])
    for m in moments:
        assert 15 - 1e-6 <= m.duration_sec <= 60 + 1e-6


def test_tail_window_of_zero_length_is_ignored():
    windows = _windows([90] * 6) + [WindowScore(30.0, 30.0, 0, 0.0)]
    moments = select_moments(windows, _settings(), 30.0)
    assert moments and moments[0].end_sec <= 30.0


def test_short_settings_respected_for_tiny_video():
    base = UserSettings()
    settings = replace(base, shorts=ShortsSettings(allowed_durations_sec=(5,), min_duration_sec=5, max_duration_sec=5),
                       viral_score=ViralScoreSettings(queue_threshold=0))
    moments = select_moments(_windows([50, 60]), settings, 10.0)
    assert moments and all(abs(m.duration_sec - 5.0) < 1e-6 for m in moments)


def _scored_window(start, motion, presence, scenes, audio, audio_detail="", events=0.0, events_detail=""):
    from core.entities.settings import ViralScoreSettings
    from cutting.clip_selector_service import WindowScore
    from scoring.viral_score_service import ScoreInputs, compute_score_breakdown

    inputs = ScoreInputs(motion, presence, scenes, audio_event=audio, motion_events=events, audio_detail=audio_detail,
                         motion_events_detail=events_detail)
    breakdown = compute_score_breakdown(inputs, ViralScoreSettings())
    return WindowScore(start, start + 5.0, breakdown.score, motion, (), inputs, breakdown)


def test_moment_breakdown_sums_to_the_moment_score_and_merges_details():
    from core.entities.settings import ShortsSettings, UserSettings, ViralScoreSettings
    from cutting.clip_selector_service import select_moments
    from dataclasses import replace

    windows = [
        _scored_window(0.0, 0.9, 1.0, 1, 0.9, "лай", 0.5, "1 прыжок"),
        _scored_window(5.0, 0.8, 1.0, 2, 1.0, "лай, смех", 0.0),
        _scored_window(10.0, 0.9, 1.0, 0, 0.7, "смех", 0.5, "1 падение"),
        _scored_window(15.0, 0.1, 0.0, 0, 0.0),
    ]
    settings = replace(UserSettings(), shorts=ShortsSettings(allowed_durations_sec=(15,), min_duration_sec=15, max_duration_sec=15),
                       viral_score=ViralScoreSettings(queue_threshold=10, relative_top_ratio=0.75))
    moments = select_moments(windows, settings, 20.0)
    moment = moments[0]
    assert moment.breakdown and sum(p.points for p in moment.breakdown) == pytest.approx(moment.viral_score, abs=1.0)
    by_key = {p.key: p for p in moment.breakdown}
    assert set(by_key) == {"motion", "presence", "scene", "audio", "motion_events"}
    assert by_key["audio"].detail.split(", ")[0] == "лай" and "смех" in by_key["audio"].detail
    assert by_key["scene"].detail == "3 смен(ы)"
    assert "1 прыжок" in by_key["motion_events"].detail and "1 падение" in by_key["motion_events"].detail
    assert all(p.points >= 0 for p in moment.breakdown)


def test_windows_without_breakdown_give_moment_without_it():
    from core.entities.settings import UserSettings

    windows = [WindowScore(0.0, 5.0, 90, 0.5), WindowScore(5.0, 10.0, 90, 0.5), WindowScore(10.0, 15.0, 90, 0.5)]
    moments = select_moments(windows, UserSettings(), 15.0)
    assert moments and moments[0].breakdown == ()
