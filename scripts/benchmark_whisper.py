"""Сравнение моделей faster-whisper на шумных моментах реального видео.

    python scripts/benchmark_whisper.py video.mp4 --starts 150 210 540 ... \
        --models small medium deepdml/faster-whisper-large-v3-turbo-ct2 large-v3

Эталонной расшифровки нет, поэтому самой сильной модели (последняя в списке)
отводится роль псевдо-эталона: считается расхождение слов (WER) остальных с ней.
Дополнительно: время, средняя лог-вероятность сегментов, доля латиницы
(признак галлюцинаций на русской речи) и доля повторов слов.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOW_SEC = 30.0


def extract_wav(video: Path, start: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{WINDOW_SEC}", "-i", str(video),
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )


_CJK = "\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af"


def _to_simplified(text: str) -> str:
    """Традиционные -> упрощённые иероглифы (если установлен zhconv): модели пишут то одним,
    то другим письмом, и без этого посимвольная метрика штрафует за письмо, а не за ошибку."""
    try:
        from zhconv import convert
    except ImportError:
        return text
    return convert(text, "zh-cn")


def normalize(text: str) -> list[str]:
    """Слова латиницы/кириллицы и ОТДЕЛЬНЫЕ иероглифы (для CJK метрика получается посимвольной, CER)."""
    text = _to_simplified(text)
    return re.findall(rf"[a-zа-яё0-9]+|[{_CJK}]", text.lower().replace("ё", "е"))


def wer(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        cur = [i]
        for j, h in enumerate(hypothesis, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--starts", nargs="+", type=float, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", default=str(ROOT / "docs" / "whisper_benchmark_raw.json"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--language", default="ru", help='"ru" или "auto" (автоопределение языка окна)')
    parser.add_argument("--tag", default="", help="суффикс ключа результата, например @auto")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    video = Path(args.video).expanduser()
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    with tempfile.TemporaryDirectory() as tmp:
        wavs = {}
        for start in args.starts:
            wavs[start] = Path(tmp) / f"w{int(start)}.wav"
            extract_wav(video, start, wavs[start])

        for name in args.models:
            entry = results.setdefault(name + args.tag, {"windows": {}})
            entry["language"] = args.language
            entry["cpu_threads"] = args.threads
            t0 = time.time()
            model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=args.threads, local_files_only=True)
            entry["load_sec"] = round(time.time() - t0, 1)
            print(f"[{name}] загружена за {entry['load_sec']} с", flush=True)
            for start in args.starts:
                key = str(int(start))
                t0 = time.time()
                segments, info = model.transcribe(str(wavs[start]), word_timestamps=True, language=None if args.language == "auto" else args.language, vad_filter=True)
                segments = list(segments)
                sec = time.time() - t0
                words = [w.word.strip() for s in segments for w in (s.words or [])]
                logprobs = [s.avg_logprob for s in segments]
                entry["windows"][key] = {
                    "detected_language": getattr(info, "language", None),
                    "sec": round(sec, 1),
                    "text": " ".join(words),
                    "n_words": len(words),
                    "avg_logprob": round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
                }
                print(f"[{name}] {key:>5s} {sec:5.1f}s {len(words):3d} слов: {' '.join(words)[:90]}", flush=True)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            del model

    reference_name = args.models[-1] + args.tag
    for name, entry in results.items():
        if name == reference_name or reference_name not in results:
            continue
        for key, window in entry["windows"].items():
            ref = results[reference_name]["windows"].get(key)
            if ref:
                window["wer_vs_reference"] = round(wer(normalize(ref["text"]), normalize(window["text"])), 2)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute(out_path: Path, reference_name: str) -> None:
    """Пересчитать WER к эталону по сохранённым текстам (без повторной транскрипции)."""
    results = json.loads(out_path.read_text())
    for name, entry in results.items():
        if name == reference_name:
            continue
        for key, window in entry["windows"].items():
            ref = results[reference_name]["windows"].get(key)
            if ref:
                window["wer_vs_reference"] = round(wer(normalize(ref["text"]), normalize(window["text"])), 2)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
