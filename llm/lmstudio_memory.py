"""Сколько памяти держат процессы LM Studio (llama-server и др.) — по /proc.

Приложение работает в distrobox-контейнере с общим с хостом пространством процессов, поэтому процессы LM Studio
(Flatpak на хосте) видны. Считается выделенная память: RssAnon + VmSwap по дереву процессов от главного
`lm-studio` (без отображённых файлов весов — их ядро может вытеснить само, и без GTT iGPU — она не меняется
между запросами). Именно она растёт от кэша промптов llama-server (`--cache-ram`, по умолчанию до 8 ГиБ):
каждый запрос с новым кадром сохраняет своё KV-состояние (замер: +0.4–0.6 ГиБ на клип, docs/lmstudio_memory.md).
Процессов не видно (другое окружение) — None.
"""

from __future__ import annotations

from pathlib import Path

PROC = Path("/proc")


def processes(proc: Path = PROC) -> dict[int, tuple[int, str, str]]:
    """pid -> (ppid, comm, cmdline)."""
    result = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            comm = stat[stat.index("(") + 1: stat.rindex(")")]
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            result[int(entry.name)] = (ppid, comm, cmd)
        except (OSError, ValueError, IndexError):
            continue
    return result


def lmstudio_tree(procs: dict[int, tuple[int, str, str]]) -> set[int]:
    """Главный процесс LM Studio (не --type=... дочерний Electron) и все его потомки."""
    roots = {pid for pid, (ppid, comm, cmd) in procs.items()
             if "lm-studio" in comm and "--type=" not in cmd and procs.get(ppid, (0, "", ""))[1] != comm}
    tree, changed = set(roots), True
    while changed:
        changed = False
        for pid, (ppid, _, _) in procs.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def process_memory_kib(pid: int, proc: Path = PROC) -> tuple[int, int, int]:
    """(RssAnon, RssFile + RssShmem, VmSwap) в КиБ."""
    anon = file = swap = 0
    try:
        for line in (proc / str(pid) / "status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                anon = int(line.split()[1])
            elif line.startswith(("RssFile:", "RssShmem:")):
                file += int(line.split()[1])
            elif line.startswith("VmSwap:"):
                swap = int(line.split()[1])
    except OSError:
        pass
    return anon, file, swap


def lmstudio_private_mib(proc: Path = PROC) -> float | None:
    """RssAnon + VmSwap всех процессов LM Studio, МиБ; None — процессов LM Studio не видно."""
    tree = lmstudio_tree(processes(proc))
    if not tree:
        return None
    total = 0
    for pid in tree:
        anon, _, swap = process_memory_kib(pid, proc)
        total += anon + swap
    return total / 1024
