"""Фоновая музыка клипа с приглушением на речи (пункт ТЗ «МУЗЫКА»).

Треки берутся из assets/music/ (кладёт пользователь; в репозиторий музыка не добавляется —
чужой трек даст претензию Content ID на YouTube). Папка пуста — клип остаётся без музыки,
в лог пишется причина. Оригинальный звук клипа (в том числе звуки животных) сохраняется:
приглушается только МУЗЫКА, и только на отрезках, где Whisper нашёл речь (а не по громкости
всего звука — иначе лай или смех тоже «выключали» бы музыку). Видеопоток не перекодируется.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

from loguru import logger

from core.entities.subtitle import WordTiming
from core.exceptions import FFmpegExecutionError

MUSIC_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}
SPEECH_GAP_SEC = 0.6      # паузы между словами короче этого не считаются концом речи
DUCK_PAD_SEC = 0.15       # музыка затихает чуть ДО первого слова и возвращается чуть ПОСЛЕ последнего
DUCK_RAMP_SEC = 0.25      # плавность спада/возврата громкости
FADE_IN_SEC = 0.5
FADE_OUT_SEC = 1.0


def list_tracks(library: Path) -> list[Path]:
    if not library.is_dir():
        return []
    return sorted(p for p in library.iterdir() if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def pick_track(tracks: list[Path], key: str, clip_duration_sec: float, durations: dict[Path, float] | None = None) -> Path | None:
    """Стабильный (по имени клипа) выбор трека; тому, что не короче клипа, отдаётся предпочтение — без зацикливания."""
    if not tracks:
        return None
    durations = durations or {t: probe_duration(t) for t in tracks}
    long_enough = [t for t in tracks if durations.get(t, 0.0) >= clip_duration_sec]
    pool = long_enough or tracks
    return pool[zlib.crc32(key.encode("utf-8")) % len(pool)]


def speech_intervals(words: list[WordTiming], clip_duration_sec: float) -> list[tuple[float, float]]:
    """Отрезки речи (от начала клипа) из слов Whisper: слова с паузой < SPEECH_GAP_SEC склеиваются."""
    intervals: list[list[float]] = []
    for word in sorted(words, key=lambda w: w.start_sec):
        if intervals and word.start_sec - intervals[-1][1] <= SPEECH_GAP_SEC:
            intervals[-1][1] = max(intervals[-1][1], word.end_sec)
        else:
            intervals.append([word.start_sec, word.end_sec])
    return [(max(0.0, a - DUCK_PAD_SEC), min(clip_duration_sec, b + DUCK_PAD_SEC)) for a, b in intervals]


def build_duck_expression(intervals: list[tuple[float, float]], duck_db: float, ramp_sec: float = DUCK_RAMP_SEC) -> str | None:
    """Выражение ffmpeg для volume=...:eval=frame: 1 вне речи, 10^(duck_db/20) на речи, плавные переходы."""
    if not intervals:
        return None
    gain = 10 ** (min(0.0, duck_db) / 20)
    r = ramp_sec
    shapes = [f"clip(min((t-{a - r:.3f})/{r},({b + r:.3f}-t)/{r}),0,1)" for a, b in intervals]
    combined = shapes[0]
    for shape in shapes[1:]:
        combined = f"max({combined},{shape})"
    return f"1-{1 - gain:.4f}*{combined}"


def _has_audio(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip())


def mix_music(
    video_path: Path,
    music_path: Path,
    clip_duration_sec: float,
    volume: float,
    duck_intervals: list[tuple[float, float]],
    duck_db: float,
    audio_bitrate_kbps: int = 192,
    audio_codec: str = "aac",
) -> None:
    """Подмешивает музыку в звук video_path (файл заменяется). Видео копируется без перекодирования."""
    music_chain = [
        f"volume={max(0.0, volume):.4f}",
    ]
    expression = build_duck_expression(duck_intervals, duck_db)
    if expression is not None:
        music_chain.append(f"volume='{expression}':eval=frame")
    fade_out_start = max(0.0, clip_duration_sec - FADE_OUT_SEC)
    music_chain += [f"afade=t=in:st=0:d={FADE_IN_SEC}", f"afade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_SEC}"]
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"

    with tempfile.TemporaryDirectory(prefix="memo_cat_music_") as tmp_dir:
        mixed = Path(tmp_dir) / f"mixed{video_path.suffix}"
        if _has_audio(video_path):
            graph = (
                f"[1:a]{fmt},{','.join(music_chain)}[m];[0:a]{fmt}[o];"
                f"[o][m]amix=inputs=2:duration=first:normalize=0:dropout_transition=0[a]"
            )
        else:   # у клипа нет своего звука — остаётся одна музыка
            graph = f"[1:a]{fmt},{','.join(music_chain)}[a]"
        command = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", audio_codec, "-b:a", f"{audio_bitrate_kbps}k",
            "-t", f"{clip_duration_sec:.3f}", "-movflags", "+faststart", str(mixed),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise FFmpegExecutionError(command, result.returncode, result.stderr)
        shutil.move(str(mixed), str(video_path))
    logger.info("Музыка «{}» добавлена в {} (громкость {:.2f}, приглушение на речи: {} отрезков)",
                music_path.name, video_path.name, volume, len(duck_intervals))
