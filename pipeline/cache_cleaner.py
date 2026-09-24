"""Очистка кэша при нехватке места на диске (пункт ТЗ «КЭШ»).

Безопасные шаги (делаются сами, по возрастающей):
  1. временные файлы приложения: недоделанные `_tmp_audio_*.wav` / `_tmp_subs_*.ass` в папке экспорта и
     каталоги `memo_cat_*` во временной папке системы — остатки упавших запусков;
  2. ротированные логи старше log_keep_days (активный logs/app.log не трогается);
  3. все ротированные логи.
Готовые клипы, обложки, JSON, SRT и ASS в export/output НИКОГДА не удаляются без подтверждения
пользователя: если места всё равно мало, cleaner лишь ПРЕДЛАГАЕТ самые старые результаты
(result_groups), а delete_result_groups() без confirmed=True отказывается что-либо удалять.
Свежие временные файлы (моложе min_temp_age_sec) не трогаются: они могут принадлежать идущему сейчас видео.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger

APP_TEMP_DIR_PREFIXES = ("memo_cat_banner_", "memo_cat_export_", "memo_cat_music_")
OUTPUT_TEMP_GLOBS = ("_tmp_audio_*.wav", "_tmp_subs_*.ass")
ACTIVE_LOG_NAME = "app.log"
RESULT_SUFFIXES = (".mp4", "_cover.png", ".json", ".srt", ".ass")   # что входит в «результат» клипа


@dataclass
class CleanupReport:
    free_before: int
    free_after: int
    removed: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    still_low: bool = False


@dataclass(frozen=True)
class ResultGroup:
    """Один готовый клип со всеми файлами рядом (клип, обложка, JSON, субтитры)."""

    stem: str
    files: tuple[Path, ...]
    size_bytes: int
    mtime: float


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
        return path.stat().st_size
    except OSError:
        return 0


class CacheCleaner:
    def __init__(
        self,
        output_dir: Path,
        logs_dir: Path,
        min_free_bytes: int,
        log_keep_days: int = 14,
        min_temp_age_sec: float = 3600.0,
        temp_root: Path | None = None,
        disk_free: Callable[[Path], int] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._output_dir = output_dir
        self._logs_dir = logs_dir
        self._min_free = min_free_bytes
        self._log_keep_sec = log_keep_days * 86400
        self._min_temp_age = min_temp_age_sec
        self._temp_root = temp_root or Path(tempfile.gettempdir())
        self._disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        self._now = now

    # ------------------------------------------------------------------ состояние диска
    def free_bytes(self) -> int:
        probe = self._output_dir
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return self._disk_free(probe)

    def is_low_space(self) -> bool:
        return self.free_bytes() < self._min_free

    # ------------------------------------------------------------------ безопасная очистка
    def clean_if_low(self, temp_age_sec: float | None = None) -> CleanupReport:
        """Если места мало — удаляет временные файлы, затем старые логи, пока не хватит. Клипы не трогает.
        temp_age_sec: порог возраста временных файлов (по умолчанию min_temp_age_sec; при пустой очереди можно меньше)."""
        report = CleanupReport(free_before=self.free_bytes(), free_after=0)
        report.free_after = report.free_before
        if report.free_before >= self._min_free:
            return report

        age = self._min_temp_age if temp_age_sec is None else temp_age_sec
        steps: list[Callable[[], list[Path]]] = [
            lambda: self._stale_temp(age),
            lambda: self._rotated_logs(older_than_sec=self._log_keep_sec),
            lambda: self._rotated_logs(older_than_sec=0),
        ]
        for step in steps:
            for path in step():
                freed = _size(path)
                try:
                    shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
                except OSError as exc:
                    logger.warning("Кэш: не удалось удалить {}: {}", path, exc)
                    continue
                report.removed.append(path)
                report.freed_bytes += freed
            report.free_after = self.free_bytes()
            if report.free_after >= self._min_free:
                break
        report.still_low = report.free_after < self._min_free
        logger.info(
            "Кэш: мало места ({:.1f} ГБ), удалено {} файл(ов), освобождено {:.1f} МБ, теперь свободно {:.1f} ГБ{}",
            report.free_before / 1e9, len(report.removed), report.freed_bytes / 1e6, report.free_after / 1e9,
            " — этого мало, нужно подтверждение на удаление клипов" if report.still_low else "",
        )
        return report

    def _stale_temp(self, min_age_sec: float) -> list[Path]:
        cutoff = self._now() - min_age_sec
        found: list[Path] = []
        if self._output_dir.is_dir():
            for pattern in OUTPUT_TEMP_GLOBS:
                found += [p for p in self._output_dir.glob(pattern) if p.is_file() and not p.is_symlink()]
        if self._temp_root.is_dir():
            for prefix in APP_TEMP_DIR_PREFIXES:
                found += [p for p in self._temp_root.glob(f"{prefix}*") if p.is_dir() and not p.is_symlink()]
        return [p for p in found if p.stat().st_mtime <= cutoff]

    def _rotated_logs(self, older_than_sec: float) -> list[Path]:
        if not self._logs_dir.is_dir():
            return []
        cutoff = self._now() - older_than_sec
        return [
            p for p in self._logs_dir.iterdir()
            if p.is_file() and not p.is_symlink() and p.name != ACTIVE_LOG_NAME
            and (".log" in p.name) and p.stat().st_mtime <= cutoff
        ]

    # ------------------------------------------------------------------ результаты: только с подтверждением
    def result_groups(self) -> list[ResultGroup]:
        """Готовые клипы (со всем, что лежит рядом), самые старые первыми — то, что МОЖНО предложить удалить."""
        if not self._output_dir.is_dir():
            return []
        groups: list[ResultGroup] = []
        for clip in self._output_dir.glob("*.mp4"):
            if not clip.is_file() or clip.is_symlink():
                continue
            files = [clip] + [
                clip.with_name(clip.stem + suffix) for suffix in RESULT_SUFFIXES[1:]
                if clip.with_name(clip.stem + suffix).is_file()
            ]
            groups.append(ResultGroup(clip.stem, tuple(files), sum(_size(f) for f in files), clip.stat().st_mtime))
        return sorted(groups, key=lambda g: g.mtime)

    def groups_to_free(self, needed_bytes: int) -> list[ResultGroup]:
        """Самые старые клипы, которых хватит, чтобы освободить needed_bytes (для показа пользователю)."""
        chosen: list[ResultGroup] = []
        total = 0
        for group in self.result_groups():
            if total >= needed_bytes:
                break
            chosen.append(group)
            total += group.size_bytes
        return chosen

    def delete_result_groups(self, groups: list[ResultGroup], confirmed: bool) -> int:
        """Удаляет клипы ТОЛЬКО при confirmed=True (пользователь подтвердил). Возвращает число освобождённых байт."""
        if not confirmed:
            raise PermissionError("Готовые клипы удаляются только с подтверждения пользователя")
        output_root = self._output_dir.resolve()
        freed = 0
        for group in groups:
            for file in group.files:
                if file.resolve().parent != output_root or file.is_symlink():   # только файлы прямо в папке экспорта
                    logger.warning("Кэш: пропущен файл вне папки экспорта: {}", file)
                    continue
                size = _size(file)
                try:
                    file.unlink()
                    freed += size
                except OSError as exc:
                    logger.warning("Кэш: не удалось удалить {}: {}", file, exc)
        logger.info("Кэш: по подтверждению пользователя удалено {} клип(ов), {:.1f} МБ", len(groups), freed / 1e6)
        return freed
