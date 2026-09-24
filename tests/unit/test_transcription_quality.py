from types import SimpleNamespace

from core.entities.subtitle import WordTiming
from subtitles import subtitle_service
from subtitles.subtitle_service import WhisperTranscriber, is_hallucination, should_drop_segment
from subtitles.translator import (
    parse_numbered,
    spread_words,
    split_into_units,
    translate_words,
)


def test_known_hallucinations_detected():
    assert is_hallucination("Субтитры сделал DimaTorzok")
    assert is_hallucination("Редактор субтитров А.Семкин Корректор А.Егорова")
    assert is_hallucination("请不吝点赞 订阅 转发")
    assert not is_hallucination("ты что, встал?")


def test_low_confidence_and_no_speech_segments_dropped():
    assert should_drop_segment("бла бла", avg_logprob=-2.0, no_speech_prob=0.1)
    assert should_drop_segment("привет", avg_logprob=-1.2, no_speech_prob=0.8)
    assert not should_drop_segment("привет", avg_logprob=-0.6, no_speech_prob=0.1)
    assert not should_drop_segment("привет", avg_logprob=-1.2, no_speech_prob=0.2)


def _fake_model(segments, language="zh", probability=0.9):
    class FakeModel:
        def transcribe(self, path, **kwargs):
            self.kwargs = kwargs
            return iter(segments), SimpleNamespace(language=language, language_probability=probability)

    return FakeModel()


def _segment(text, start, logprob=-0.5, no_speech=0.1):
    words = [SimpleNamespace(word=t, start=start + i * 0.3, end=start + i * 0.3 + 0.25) for i, t in enumerate(text.split())]
    return SimpleNamespace(text=text, words=words, avg_logprob=logprob, no_speech_prob=no_speech)


def test_transcriber_filters_hallucinations_and_reports_language(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    transcriber = WhisperTranscriber()
    model = _fake_model([
        _segment("你 是不是 在", 0.0),
        _segment("Субтитры сделал DimaTorzok", 5.0),
        _segment("мусор мусор", 8.0, logprob=-2.5),
    ])
    transcriber._model = model
    words = transcriber.transcribe(audio)
    assert [w.text for w in words] == ["你", "是不是", "在"]
    assert transcriber.last_language == "zh"
    assert model.kwargs["language"] is None          # автоопределение по умолчанию


def test_auto_language_string_means_none(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    transcriber = WhisperTranscriber()
    transcriber._model = _fake_model([])
    transcriber.transcribe(audio, language="auto")
    assert transcriber._model.kwargs["language"] is None
    transcriber.transcribe(audio, language="ru")
    assert transcriber._model.kwargs["language"] == "ru"


def _words(*items):
    return [WordTiming(t, s, e) for t, s, e in items]


def test_units_split_on_pauses_and_length():
    words = _words(("你", 0.0, 0.3), ("好", 0.4, 0.7), ("我", 3.0, 3.3), ("是", 3.4, 3.7))
    units = split_into_units(words)
    assert [[w.text for w in u] for u in units] == [["你", "好"], ["我", "是"]]


def test_parse_numbered_requires_exact_count():
    assert parse_numbered("1. Привет\n2) Как дела", 2) == ["Привет", "Как дела"]
    assert parse_numbered("1. Привет", 2) is None
    assert parse_numbered("Привет мир", 1) is None


def test_spread_words_covers_interval_proportionally():
    result = spread_words("я не знаю", 10.0, 12.0)
    assert [w.text for w in result] == ["я", "не", "знаю"]
    assert result[0].start_sec == 10.0 and abs(result[-1].end_sec - 12.0) < 1e-9
    assert all(a.end_sec <= b.start_sec + 1e-9 for a, b in zip(result, result[1:]))
    assert (result[2].end_sec - result[2].start_sec) > (result[0].end_sec - result[0].start_sec)


class _Provider:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    def complete(self, system, user, max_tokens=512, images=None):
        self.prompts.append(user)
        if self.error:
            raise self.error
        return self.answer

    def list_models(self):
        return []


def test_translate_words_keeps_timing():
    words = _words(("你", 0.0, 0.3), ("是不是", 0.4, 0.9), ("在", 1.0, 1.3), ("哎", 5.0, 5.3), ("呦", 5.4, 5.7))
    provider = _Provider("1. Ты в порядке\n2. Ой-ой")
    result = translate_words(provider, words, "ru")
    assert [w.text for w in result] == ["Ты", "в", "порядке", "Ой-ой"]
    assert result[0].start_sec == 0.0 and abs(result[2].end_sec - 1.3) < 1e-9
    assert result[3].start_sec == 5.0
    assert "你是不是在" in provider.prompts[0]        # китайские слова склеены без пробелов


def test_translate_failure_returns_none_so_original_is_kept():
    words = _words(("你", 0.0, 0.3))
    assert translate_words(_Provider(error=RuntimeError("down")), words) is None
    assert translate_words(_Provider("нет нумерации"), words) is None
    assert translate_words(_Provider(""), []) == []


def test_no_speech_after_vad_is_empty_result_not_error(tmp_path):
    """faster-whisper 1.0.x бросает ValueError('max() iterable argument is empty') на пустом
    после VAD аудио при автоопределении языка — это отсутствие речи."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    class EmptyAudioModel:
        def transcribe(self, path, **kwargs):
            raise ValueError("max() iterable argument is empty")

    transcriber = WhisperTranscriber()
    transcriber._model = EmptyAudioModel()
    assert transcriber.transcribe(audio) == []
    assert transcriber.last_language is None


def test_other_value_errors_still_reported(tmp_path):
    import pytest

    from subtitles.subtitle_service import TranscriptionError

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    class BrokenModel:
        def transcribe(self, path, **kwargs):
            raise ValueError("bad audio")

    transcriber = WhisperTranscriber()
    transcriber._model = BrokenModel()
    with pytest.raises(TranscriptionError):
        transcriber.transcribe(audio)
