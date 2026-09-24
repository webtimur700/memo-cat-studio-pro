from core.entities.subtitle import SubtitleSegment, WordTiming
from subtitles.ass_renderer import _seconds_to_ass_timestamp, _seconds_to_srt_timestamp, render_ass, render_srt
from subtitles.subtitle_service import group_words_into_segments


def test_timestamp_conversions():
    assert _seconds_to_ass_timestamp(1.5) == "0:00:01.50"
    assert _seconds_to_ass_timestamp(65.25) == "0:01:05.25"
    assert _seconds_to_srt_timestamp(1.5) == "00:00:01,500"


def test_render_ass_contains_karaoke_tags_and_falls_back_gracefully():
    segments = [SubtitleSegment(words=[WordTiming("Кот", 1.5, 1.8), WordTiming("прыгнул", 1.8, 2.3)])]

    ass_output = render_ass(segments, style_preset="neon_pop")
    assert "[Script Info]" in ass_output
    assert "Poppins ExtraBold" in ass_output
    assert "Dialogue: 0,0:00:01.50,0:00:02.30" in ass_output

    fallback_output = render_ass(segments, style_preset="unknown_style")
    assert "Montserrat ExtraBold" in fallback_output


def test_render_srt():
    segments = [SubtitleSegment(words=[WordTiming("Кот", 1.5, 1.8), WordTiming("прыгнул", 1.8, 2.3)])]
    srt_output = render_srt(segments)
    assert "1\n00:00:01,500 --> 00:00:02,300\nКот прыгнул" in srt_output


def test_group_words_into_segments_by_word_limit():
    words = [
        WordTiming("Кот", 0.0, 0.3), WordTiming("прыгнул", 0.3, 0.7),
        WordTiming("на", 0.7, 0.8), WordTiming("шкаф", 0.8, 1.2),
        WordTiming("и", 1.2, 1.3), WordTiming("упал", 1.3, 1.7),
    ]
    segments = group_words_into_segments(words, max_words_per_segment=4)
    assert [s.text for s in segments] == ["Кот прыгнул на шкаф", "и упал"]


def test_group_words_breaks_on_pause():
    words = [
        WordTiming("Привет", 0.0, 0.4), WordTiming("всем", 0.4, 0.7),
        WordTiming("сегодня", 2.0, 2.4), WordTiming("дождь", 2.4, 2.7),
    ]
    segments = group_words_into_segments(words)
    assert [s.text for s in segments] == ["Привет всем", "сегодня дождь"]


def test_timestamps_never_overflow_the_fraction_field():
    # раньше 1.9996 с давало «00:00:01,1000» — такой SRT плееры и YouTube отвергают
    assert _seconds_to_srt_timestamp(1.9996) == "00:00:02,000"
    assert _seconds_to_srt_timestamp(3599.9999) == "01:00:00,000"
    assert _seconds_to_ass_timestamp(1.996) == "0:00:02.00"
    assert _seconds_to_ass_timestamp(-0.4) == "0:00:00.00"


def test_srt_cues_are_numbered_and_separated_by_blank_line():
    segments = [
        SubtitleSegment(words=[WordTiming("раз", 0.0, 0.5)]),
        SubtitleSegment(words=[WordTiming("два", 1.0, 1.5)]),
    ]
    assert render_srt(segments) == "1\n00:00:00,000 --> 00:00:00,500\nраз\n\n2\n00:00:01,000 --> 00:00:01,500\nдва\n"
