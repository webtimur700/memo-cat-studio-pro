"""Звуковые события для Viral Score: лай, мяуканье, мурлыканье, смех и т.п. (вес audio_event).

Офлайн-классификатор YAMNet (Google, Apache-2.0; обучен на AudioSet, 521 класс), ONNX-версия из
models/yamnet/ через onnxruntime на CPU. Он смотрит на 0.96-секундные окна звука с шагом 0.48 с
и для каждого выдаёт вероятности классов. Нас интересуют классы животных и смеха — по имени
класса из yamnet_class_map.csv, а не по номеру. Модель скачивает scripts/download_models.py.

Скорость на CPU: ~0.01 с на 10 с звука, то есть весь получасовой ролик анализируется за пару секунд.
Звук декодируется ffmpeg'ом порциями по минуте, память ограничена независимо от длины видео.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

SAMPLE_RATE = 16000
FRAME_HOP_SEC = 0.48          # шаг окон YAMNet
FRAME_LEN_SEC = 0.96          # длина окна YAMNet
HOP_SAMPLES = int(FRAME_HOP_SEC * SAMPLE_RATE)
FRAME_SAMPLES = int(FRAME_LEN_SEC * SAMPLE_RATE)
FRAMES_PER_CHUNK = 125        # 60 секунд звука за один вызов модели

# Классы AudioSet и вес каждого: насколько «вирусным» считаем такой звук. Имена — из yamnet_class_map.csv.
EVENT_CLASS_WEIGHTS: dict[str, float] = {
    "Bark": 1.0, "Yip": 1.0, "Howl": 0.8, "Growling": 0.8, "Whimper (dog)": 0.8, "Dog": 0.7,
    "Meow": 1.0, "Purr": 0.9, "Caterwaul": 1.0, "Hiss": 0.9, "Cat": 0.7,
    "Laughter": 1.0, "Baby laughter": 1.0, "Giggle": 1.0, "Belly laugh": 1.0, "Chuckle, chortle": 0.9,
    "Snicker": 0.9, "Roaring cats (lions, tigers)": 0.7, "Squeal": 0.5, "Bird vocalization, bird call, bird song": 0.4,
}
# как называть звук пользователю (в объяснении оценки момента)
CLASS_LABELS_RU: dict[str, str] = {
    "Bark": "лай", "Yip": "тявканье", "Howl": "вой", "Growling": "рычание", "Whimper (dog)": "скулёж", "Dog": "собака",
    "Meow": "мяуканье", "Purr": "мурлыканье", "Caterwaul": "кошачий вопль", "Hiss": "шипение", "Cat": "кошка",
    "Laughter": "смех", "Baby laughter": "детский смех", "Giggle": "хихиканье", "Belly laugh": "хохот",
    "Chuckle, chortle": "смешок", "Snicker": "смешок", "Roaring cats (lions, tigers)": "рёв кошачьих",
    "Squeal": "визг", "Bird vocalization, bird call, bird song": "птичье пение",
}
# вероятность, при которой звук считается «явным событием» (YAMNet редко даёт больше 0.7-0.8)
SATURATION_PROB = 0.5


@dataclass(frozen=True, slots=True)
class AudioEventTimeline:
    """Оценка «событийности» звука по времени: frame_scores[k] — 0..1 для окна YAMNet k (центр k*0.48+0.48 с)."""

    frame_scores: np.ndarray
    top_classes: tuple[str, ...]   # для каждого окна название самого вероятного интересного класса ("" — нет)

    def score_between(self, start_sec: float, end_sec: float) -> float:
        """Оценка интервала: среднее трёх сильнейших окон (одно яркое «мяу» уже событие, а фон — нет)."""
        if self.frame_scores.size == 0:
            return 0.0
        first = max(0, int((start_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC))
        last = min(self.frame_scores.size, int(np.ceil((end_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC)) + 1)
        window = self.frame_scores[first:last]
        if window.size == 0:
            return 0.0
        return float(np.sort(window)[-3:].mean())

    def labels_between(self, start_sec: float, end_sec: float, min_score: float = 0.3, limit: int = 2) -> str:
        """Какие звуки слышны в интервале: «лай, смех» (самые частые среди заметных окон; пусто — событий нет)."""
        first = max(0, int((start_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC))
        last = min(self.frame_scores.size, int(np.ceil((end_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC)) + 1)
        counts: dict[str, float] = {}
        for k in range(first, last):
            if self.frame_scores[k] >= min_score and self.top_classes[k]:
                label = CLASS_LABELS_RU.get(self.top_classes[k], self.top_classes[k])
                counts[label] = counts.get(label, 0.0) + float(self.frame_scores[k])
        return ", ".join(sorted(counts, key=lambda k: -counts[k])[:limit])

    def dominant_class_between(self, start_sec: float, end_sec: float) -> str:
        first = max(0, int((start_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC))
        last = min(self.frame_scores.size, int(np.ceil((end_sec - FRAME_LEN_SEC / 2) / FRAME_HOP_SEC)) + 1)
        if last <= first:
            return ""
        best = first + int(np.argmax(self.frame_scores[first:last]))
        return self.top_classes[best] if self.frame_scores[best] > 0.05 else ""


def load_class_map(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [row["display_name"] for row in csv.DictReader(fh)]


def decode_audio_chunks(video_path: Path, chunk_samples: int, overlap_samples: int):
    """Моно 16 кГц float32 из видео (ffmpeg -> pipe) порциями chunk_samples; соседние порции перекрываются на
    overlap_samples, чтобы окна YAMNet не терялись на стыках. Видео без звука даёт пустой результат."""
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    step = chunk_samples - overlap_samples
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buffer = np.zeros(0, dtype=np.float32)
    try:
        while True:
            data = process.stdout.read(4 * step)
            if not data:
                break
            buffer = np.concatenate([buffer, np.frombuffer(data[: len(data) // 4 * 4], dtype=np.float32)])
            while buffer.size >= chunk_samples:
                yield buffer[:chunk_samples]
                buffer = buffer[step:]
        if buffer.size > overlap_samples or (buffer.size and overlap_samples == 0):
            yield buffer
    finally:
        process.stdout.close()
        process.wait()


class AudioEventClassifier:
    def __init__(self, model_dir: Path, session=None) -> None:
        self._class_names = load_class_map(model_dir / "yamnet_class_map.csv")
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2   # фоновая работа: не отнимаем все ядра у видео
            session = ort.InferenceSession(str(model_dir / "yamnet.onnx"), options, providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name
        indices = [i for i, name in enumerate(self._class_names) if name in EVENT_CLASS_WEIGHTS]
        self._event_indices = np.array(indices, dtype=np.int64)
        self._event_weights = np.array([EVENT_CLASS_WEIGHTS[self._class_names[i]] for i in indices], dtype=np.float32)
        self._event_names = [self._class_names[i] for i in indices]

    @classmethod
    def load(cls, model_dir: Path) -> "AudioEventClassifier | None":
        """None, если модели нет или она не загрузилась: скоринг работает без звуковых событий (честная деградация)."""
        if not (model_dir / "yamnet.onnx").is_file() or not (model_dir / "yamnet_class_map.csv").is_file():
            logger.warning("YAMNet не найден ({}) — Viral Score без звуковых событий. См. scripts/download_models.py", model_dir)
            return None
        try:
            return cls(model_dir)
        except Exception as exc:
            logger.warning("Не удалось загрузить YAMNet ({}): {} — Viral Score без звуковых событий", model_dir, exc)
            return None

    def frame_probabilities(self, waveform: np.ndarray) -> np.ndarray:
        """(число окон, 521) вероятности классов для звука любой длины >= 0.96 с (короче — дополняется тишиной)."""
        if waveform.size < FRAME_SAMPLES:
            waveform = np.pad(waveform, (0, FRAME_SAMPLES - waveform.size))
        return self._session.run(None, {self._input_name: waveform.astype(np.float32)})[0]

    def _frame_event_scores(self, probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
        weighted = probabilities[:, self._event_indices] * self._event_weights   # (окна, интересные классы)
        best = weighted.argmax(axis=1)
        scores = np.clip(weighted.max(axis=1) / SATURATION_PROB, 0.0, 1.0)
        return scores, [self._event_names[i] for i in best]

    def analyze_video(self, video_path: Path) -> AudioEventTimeline:
        chunk_samples = (FRAMES_PER_CHUNK - 1) * HOP_SAMPLES + FRAME_SAMPLES
        overlap = FRAME_SAMPLES - HOP_SAMPLES
        scores: list[np.ndarray] = []
        names: list[str] = []
        for chunk in decode_audio_chunks(video_path, chunk_samples, overlap):
            chunk_scores, chunk_names = self._frame_event_scores(self.frame_probabilities(chunk))
            scores.append(chunk_scores)
            names.extend(chunk_names)
        if not scores:
            logger.info("В {} нет звуковой дорожки — звуковые события не оцениваются", video_path.name)
            return AudioEventTimeline(np.zeros(0, dtype=np.float32), ())
        return AudioEventTimeline(np.concatenate(scores), tuple(names))
