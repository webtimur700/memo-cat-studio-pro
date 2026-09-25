"""Фоновая музыка клипа с приглушением на речи (пункт ТЗ «МУЗЫКА»).

Громкость музыки задаётся не множителем, а смещением в дБ ниже оригинального звука клипа: громкость клипа
и трека измеряются (ffmpeg ebur128, LUFS), трек подтягивается так, чтобы звучать на music_offset_db ниже
оригинала, а на речи приглушается ещё сильнее (duck_level_db). Тихий клип получает тихую музыку, громкий —
громче, а трек с любой мастеринговой громкостью ложится на один и тот же уровень относительно клипа.

Треки берутся из assets/music/ (кладёт пользователь; в репозиторий музыка не добавляется —
чужой трек даст претензию Content ID на YouTube). Папка пуста — клип остаётся без музыки,
в лог пишется причина. Оригинальный звук клипа (в том числе звуки животных) сохраняется:
приглушается только МУЗЫКА, и только на отрезках, где Whisper нашёл речь (а не по громкости
всего звука — иначе лай или смех тоже «выключали» бы музыку). Видеопоток не перекодируется.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from functools import lru_cache
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

MIN_MEANINGFUL_LUFS = -50.0      # тише этого у клипа «нет звука»: смещение относительно него ничего не значит
MUSIC_ONLY_LUFS = -18.0          # уровень музыки, когда у клипа нет своего звука
MAX_GAIN_DB = 15.0               # подтягивать очень тихий трек сильнее не имеет смысла (шум)
MIN_GAIN_DB = -40.0


@dataclass(frozen=True, slots=True)
class MixReport:
    """Что измерили и какой сделали микс (пишется в JSON клипа для разбора)."""

    clip_lufs: float | None          # громкость оригинального звука клипа (None — звука нет)
    track_lufs: float                # громкость трека
    target_lufs: float               # на каком уровне звучит музыка (вне речи)
    gain_db: float                   # усиление трека
    offset_db: float                 # заданное смещение относительно оригинала
    duck_db: float

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in {
            "clip_lufs": self.clip_lufs, "track_lufs": self.track_lufs, "target_lufs": self.target_lufs,
            "gain_db": self.gain_db, "offset_db": self.offset_db, "duck_db": self.duck_db,
        }.items()}


_LUFS_RE = re.compile(r"^\s*I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", re.MULTILINE)


def measure_lufs(path: Path, start_sec: float | None = None, duration_sec: float | None = None) -> float | None:
    """Интегральная громкость звука файла (ebur128, LUFS); None — нет звуковой дорожки или тишина."""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats"]
    if start_sec is not None:
        command += ["-ss", f"{start_sec:.3f}"]
    if duration_sec is not None:
        command += ["-t", f"{duration_sec:.3f}"]
    command += ["-i", str(path), "-vn", "-af", "ebur128=framelog=quiet", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    matches = _LUFS_RE.findall(result.stderr)
    if not matches:
        return None
    value = float(matches[-1])
    return value if value > -69.0 else None      # -70 LUFS — ebur128 так пишет «тишина»


@lru_cache(maxsize=256)
def _track_lufs_cached(path: str, mtime_ns: int, duration_key: float | None) -> float | None:
    return measure_lufs(Path(path), None, duration_key)


def track_lufs(path: Path, duration_sec: float | None = None) -> float | None:
    """Громкость той части трека, что реально прозвучит (музыка всегда начинается с начала трека; трек короче
    клипа зацикливается, тогда это весь трек). Громкость по ходу трека меняется, поэтому интеграл по всему
    треку давал бы уровень на пару дБ мимо."""
    key = None if duration_sec is None else round(duration_sec * 2) / 2   # кэш по полусекундам
    return _track_lufs_cached(str(path), path.stat().st_mtime_ns, key)


def plan_gain(clip_lufs: float | None, track_loudness: float, offset_db: float) -> tuple[float, float]:
    """(усиление трека в дБ, целевой уровень музыки в LUFS): музыка на offset_db ниже оригинала клипа."""
    if clip_lufs is None or clip_lufs < MIN_MEANINGFUL_LUFS:
        target = MUSIC_ONLY_LUFS
    else:
        target = clip_lufs + min(0.0, offset_db)
    gain = max(MIN_GAIN_DB, min(MAX_GAIN_DB, target - track_loudness))
    return gain, track_loudness + gain


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
    offset_db: float,
    duck_intervals: list[tuple[float, float]],
    duck_db: float,
    audio_bitrate_kbps: int = 192,
    audio_codec: str = "aac",
) -> MixReport:
    """Подмешивает музыку в звук video_path (файл заменяется), видео копируется без перекодирования.
    Музыка звучит на offset_db ниже оригинала клипа (по LUFS), на речи — ещё на duck_db тише."""
    has_audio = _has_audio(video_path)
    clip_loudness = measure_lufs(video_path) if has_audio else None
    music_loudness = track_lufs(music_path, clip_duration_sec)
    if music_loudness is None:
        raise ValueError(f"не удалось измерить громкость трека {music_path.name}")
    gain_db, target = plan_gain(clip_loudness, music_loudness, offset_db)

    music_chain = [f"volume={gain_db:.2f}dB"]
    expression = build_duck_expression(duck_intervals, duck_db)
    if expression is not None:
        music_chain.append(f"volume='{expression}':eval=frame")
    fade_out_start = max(0.0, clip_duration_sec - FADE_OUT_SEC)
    music_chain += [f"afade=t=in:st=0:d={FADE_IN_SEC}", f"afade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_SEC}"]
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"

    with tempfile.TemporaryDirectory(prefix="memo_cat_music_") as tmp_dir:
        mixed = Path(tmp_dir) / f"mixed{video_path.suffix}"
        if has_audio:
            graph = (
                f"[1:a]{fmt},{','.join(music_chain)}[m];[0:a]{fmt}[o];"
                f"[o][m]amix=inputs=2:duration=first:normalize=0:dropout_transition=0,alimiter=limit=0.95[a]"
            )
        else:   # у клипа нет своего звука — остаётся одна музыка
            graph = f"[1:a]{fmt},{','.join(music_chain)},alimiter=limit=0.95[a]"
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
    report = MixReport(clip_loudness, music_loudness, target, gain_db, offset_db, duck_db)
    logger.info(
        "Музыка «{}» в {}: оригинал {} LUFS, трек {:.1f} LUFS, усиление {:+.1f} дБ -> музыка {:.1f} LUFS ({:+.0f} дБ от оригинала), "
        "приглушение на речи: {} отрезков",
        music_path.name, video_path.name, "нет" if clip_loudness is None else f"{clip_loudness:.1f}", music_loudness,
        gain_db, target, offset_db, len(duck_intervals),
    )
    return report
