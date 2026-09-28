"""Память LM Studio во времени: для замера роста на длинной очереди.

Раз в --interval секунд пишет строку CSV:
    time, mem_available_mib, swap_used_mib, gtt_used_mib, lms_rss_anon_mib, lms_rss_file_mib, lms_swap_mib, lms_procs
lms_* — сумма по дереву процессов LM Studio (корень — `lm-studio --run-as-service` или главный процесс
`lm-studio`, процессы хоста видны из distrobox-контейнера). RssAnon — выделенная процессом память (кэши, KV,
буферы), RssFile — отображённые файлы (веса GGUF: их ядро может вытеснить), VmSwap — вытесненная в своп анонимная
память (при нехватке RssAnon «падает», а память просто уходит в своп: считать надо anon + swap). GTT — память iGPU из общей ОЗУ
(туда уходят веса и KV при выгрузке слоёв на GPU), в RSS процесса она не видна.

    python scripts/monitor_lmstudio_memory.py out.csv [--interval 2]      запись (до Ctrl+C / SIGTERM)
    python scripts/monitor_lmstudio_memory.py out.csv --report CLIPS_DIR  память по клипам: момент готовности клипа —
                                                                          время записи его .json (после текстов LLM)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.lmstudio_memory import lmstudio_tree, process_memory_kib, processes  # noqa: E402

GTT_USED = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"))


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0])
    return values


def sample() -> list[float]:
    info = meminfo()
    tree = lmstudio_tree(processes())
    anon = file = swap = 0
    for pid in tree:
        a, f, sw = process_memory_kib(pid)
        anon += a
        file += f
        swap += sw
    gtt = sum(int(p.read_text()) for p in GTT_USED) / 2**20 if GTT_USED else 0.0
    return [round(time.time(), 1), info["MemAvailable"] // 1024, (info["SwapTotal"] - info["SwapFree"]) // 1024,
            round(gtt), anon // 1024, file // 1024, swap // 1024, len(tree)]


def report(csv_path: Path, clips_dir: Path) -> list[dict]:
    import csv

    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(open(csv_path, encoding="utf-8"))]
    clips = sorted(clips_dir.glob("*_moment*.json"), key=lambda p: p.stat().st_mtime)
    base = rows[0]
    result = []
    print(f"{'#':>3} {'клип':<24} {'+с':>6} {'свободно':>9} {'своп':>6} {'GTT':>6} {'LMS anon+swap':>14} {'Δ от старта':>11}")
    for n, clip in enumerate(clips, 1):
        t = clip.stat().st_mtime
        row = min(rows, key=lambda r: abs(r["time"] - t))
        lms = row["lms_rss_anon_mib"] + row.get("lms_swap_mib", 0)
        item = {"n": n, "clip": clip.stem, "sec": round(t - base["time"]), "mem_available_mib": row["mem_available_mib"],
                "swap_used_mib": row["swap_used_mib"], "gtt_used_mib": row["gtt_used_mib"], "lms_anon_swap_mib": lms}
        result.append(item)
        print(f"{n:>3} {clip.stem[-24:]:<24} {item['sec']:>6} {row['mem_available_mib'] / 1024:>8.1f}Г {row['swap_used_mib'] / 1024:>5.1f}Г "
              f"{row['gtt_used_mib'] / 1024:>5.1f}Г {lms / 1024:>13.2f}Г {(lms - base['lms_rss_anon_mib'] - base.get('lms_swap_mib', 0)) / 1024:>+10.2f}Г")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--report", help="папка с клипами: вместо записи — отчёт по уже записанному CSV")
    args = parser.parse_args()
    if args.report:
        report(Path(args.out), Path(args.report))
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("time,mem_available_mib,swap_used_mib,gtt_used_mib,lms_rss_anon_mib,lms_rss_file_mib,lms_swap_mib,lms_procs\n")
        while True:
            out.write(",".join(str(v) for v in sample()) + "\n")
            out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
