"""Обновление приложения из git по нажатию кнопки (пункт ТЗ «АВТООБНОВЛЕНИЕ»).

Ничего не происходит само: check() делает `git fetch` и сообщает, есть ли новые коммиты и что мешает обновлению;
update() выполняется только после подтверждения пользователя и только как fast-forward (`git merge --ff-only`), поэтому
локальные коммиты и правки не теряются и не сливаются молча. Обновлять нельзя, если в рабочей копии есть
незакоммиченные изменения отслеживаемых файлов (их перечисляем), ветка разошлась с удалённой или нет upstream.
Неотслеживаемые файлы обновлению не мешают: если новый коммит перезаписал бы такой файл, git сам откажет.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

FETCH_TIMEOUT_SEC = 60
MAX_LISTED = 20


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    branch: str = ""
    upstream: str = ""
    behind: int = 0                                  # новых коммитов на сервере
    ahead: int = 0                                   # локальных коммитов, которых нет на сервере
    new_commits: tuple[str, ...] = ()                # «abc1234 тема» (не больше MAX_LISTED)
    dirty_files: tuple[str, ...] = ()                # изменённые отслеживаемые файлы — обновлять нельзя
    untracked: int = 0
    error: str = ""                                  # git недоступен, нет сети, нет upstream...

    @property
    def can_update(self) -> bool:
        return not self.error and self.behind > 0 and self.ahead == 0 and not self.dirty_files

    @property
    def message(self) -> str:
        if self.error:
            return f"Не удалось проверить обновления: {self.error}"
        if self.dirty_files:
            shown = ", ".join(self.dirty_files[:5]) + (" и др." if len(self.dirty_files) > 5 else "")
            base = f"В рабочей копии есть незакоммиченные изменения ({len(self.dirty_files)}: {shown}) — обновление отменено, чтобы их не потерять."
            return base + (f" На сервере новых коммитов: {self.behind}." if self.behind else "")
        if self.behind == 0:
            extra = f" (локальных коммитов, которых нет на сервере: {self.ahead})" if self.ahead else ""
            return f"Установлена последняя версия ({self.branch}){extra}."
        if self.ahead:
            return (f"На сервере {self.behind} нов. коммит(ов), но ветка {self.branch} разошлась с {self.upstream} "
                    f"(локальных коммитов: {self.ahead}) — автоматическое обновление невозможно, обновите вручную (git pull --rebase).")
        return f"Доступно обновление: {self.behind} нов. коммит(ов) в {self.upstream}."


@dataclass(frozen=True, slots=True)
class UpdateResult:
    ok: bool
    message: str
    old_head: str = ""
    new_head: str = ""
    dependencies_changed: bool = False
    changed_files: tuple[str, ...] = field(default_factory=tuple)


def git_command() -> list[str] | None:
    """Как запустить git: локальный git или, если приложение работает в distrobox без git, git хоста (distrobox-host-exec)."""
    if shutil.which("git"):
        return ["git"]
    if shutil.which("distrobox-host-exec"):
        return ["distrobox-host-exec", "git"]
    return None


def _git(repo: Path, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    prefix = git_command()
    if prefix is None:
        raise FileNotFoundError("git")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}   # никаких запросов пароля, ответы на английском для разбора
    # core.askPass=true: графический запрос пароля не открывается, проверка без доступа просто завершается ошибкой
    return subprocess.run([*prefix, "-c", "core.askPass=true", *args], cwd=repo, capture_output=True, text=True, timeout=timeout, env=env, check=False)


def check_for_updates(repo: Path) -> UpdateStatus:
    """git fetch + сравнение с upstream. Ничего не меняет в рабочей копии."""
    try:
        if _git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return UpdateStatus(error="папка приложения не является git-репозиторием")
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        upstream_run = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream_run.returncode != 0:
            return UpdateStatus(branch=branch, error=f"у ветки {branch} нет удалённой ветки (upstream)")
        upstream = upstream_run.stdout.strip()

        fetch = _git(repo, "fetch", "--quiet", timeout=FETCH_TIMEOUT_SEC)
        if fetch.returncode != 0:
            return UpdateStatus(branch=branch, upstream=upstream, error=(fetch.stderr.strip() or "git fetch завершился с ошибкой")[:300])

        counts = _git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}").stdout.split()
        ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (0, 0)
        log = _git(repo, "log", "--format=%h %s", f"-{MAX_LISTED}", "HEAD..@{u}").stdout.strip().splitlines()

        status = _git(repo, "status", "--porcelain").stdout.splitlines()
        dirty = tuple(line[3:] for line in status if not line.startswith("??"))
        untracked = sum(1 for line in status if line.startswith("??"))
        return UpdateStatus(branch, upstream, behind, ahead, tuple(log), dirty, untracked)
    except FileNotFoundError:
        return UpdateStatus(error="git не найден (установите git; в distrobox — sudo dnf install git или пользуйтесь git хоста)")
    except subprocess.TimeoutExpired:
        return UpdateStatus(error="git не ответил вовремя (сеть?)")


def apply_update(repo: Path) -> UpdateResult:
    """Fast-forward до upstream. Повторно проверяет условия (между проверкой и подтверждением могло что-то измениться)."""
    status = check_for_updates(repo)
    if not status.can_update:
        return UpdateResult(False, status.message)
    old_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    merge = _git(repo, "merge", "--ff-only", "@{u}", timeout=120)
    if merge.returncode != 0:
        return UpdateResult(False, f"Не удалось обновить: {(merge.stderr or merge.stdout).strip()[:300]}", old_head, old_head)
    new_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    changed = tuple(_git(repo, "diff", "--name-only", old_head, new_head).stdout.split())
    deps = any(name in ("pyproject.toml", "requirements.txt") for name in changed)
    logger.info("Приложение обновлено: {} -> {} ({} файлов)", old_head[:7], new_head[:7], len(changed))
    note = " Изменились зависимости (pyproject.toml): после перезапуска выполните pip install -e ." if deps else ""
    return UpdateResult(True, f"Обновлено до {new_head[:7]}. Перезапустите приложение, чтобы изменения вступили в силу.{note}",
                        old_head, new_head, deps, changed)
