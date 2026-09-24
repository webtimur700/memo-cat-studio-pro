"""Проверка очереди на реальных видео: запускает настоящее MainWindow (offscreen), добавляет файлы
одним разом и печатает хронологию стадий, пик параллельности, число загрузок моделей и заголовки клипов.

    python scripts/verify_queue.py OUT_DIR video1.mp4 video2.mp4 ... [--reserve-mib N] [--abort-below-mib N]

--reserve-mib: сколько памяти оставить пайплайну при выборе LLM (по умолчанию как в приложении, 6 ГиБ).
--abort-below-mib: аварийно остановить прогон и выгрузить LLM, если MemAvailable упадёт ниже (по умолчанию 1024).

LM Studio должна быть запущена на localhost:1234. Клипы пишутся в OUT_DIR (не в export/output).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.main_window as mw  # noqa: E402
from llm import lm_studio_api as api  # noqa: E402


def mem_available_mib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def main() -> None:
    args = sys.argv[1:]
    options = {"--reserve-mib": None, "--abort-below-mib": 1024}
    for name in options:
        if name in args:
            i = args.index(name)
            options[name] = int(args[i + 1])
            del args[i:i + 2]
    out_dir = Path(args[0]).resolve()
    videos = [Path(p).resolve() for p in args[1:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    mw.EXPORT_OUTPUT_DIR = out_dir

    calls = {"llm_load": 0, "llm_unload": 0}
    orig_load, orig_unload = api.load_model, api.unload_model
    api.load_model = lambda *a, **k: (calls.__setitem__("llm_load", calls["llm_load"] + 1), orig_load(*a, **k))[1]
    api.unload_model = lambda *a, **k: (calls.__setitem__("llm_unload", calls["llm_unload"] + 1), orig_unload(*a, **k))[1]

    if options["--reserve-mib"] is not None:
        from llm.managed_provider import ManagedLMStudio

        reserve = options["--reserve-mib"]
        mw.ManagedLMStudio = lambda config: ManagedLMStudio(config, reserve_mib=reserve)

    app = QApplication([])
    window = mw.MainWindow()
    start = time.time()
    timeline: list[tuple[float, str, str]] = []
    peak_running, min_mem = 0, mem_available_mib()
    mem_start = min_mem

    orig_stage = window._on_pipeline_stage_changed

    def on_stage(job_id, stage, data):
        timeline.append((time.time() - start, job_id, stage))
        orig_stage(job_id, stage, data)

    window._on_pipeline_stage_changed = on_stage
    window.project_view.videos_added.emit(videos)
    ids = [f"job_{i}" for i in range(1, len(videos) + 1)]
    print("сразу после добавления:", {j: window.batch_view.stage_text(j) for j in ids}, flush=True)

    last_report = 0.0
    while not (window._scheduler.is_idle and not window._workers):
        QCoreApplication.processEvents()
        peak_running = max(peak_running, len(window._workers))
        min_mem = min(min_mem, mem_available_mib())
        if min_mem < options["--abort-below-mib"]:
            print(f"ПРЕРВАНО: MemAvailable {min_mem} МиБ < {options['--abort-below-mib']} — выгружаю LLM", flush=True)
            window._llm.release()
            os._exit(2)
        if time.time() - last_report > 30:
            last_report = time.time()
            print(f"[{time.time() - start:6.0f}s] " + ", ".join(f"{j}:{window.batch_view.stage_text(j)}" for j in ids), flush=True)
        time.sleep(0.05)
    total = time.time() - start
    time.sleep(3)   # дать фоновому потоку выгрузить LLM
    QCoreApplication.processEvents()

    print("\n=== итог ===")
    print(f"время: {total:.0f}s, пик одновременных задач: {peak_running}")
    print(f"MemAvailable: старт {mem_start} МиБ, минимум {min_mem} МиБ (пик падения {mem_start - min_mem} МиБ)")
    sm = window._shared_models
    print(f"загрузок YOLO: {sm.detector_loads}, Whisper: {sm.transcriber_loads}, LLM load: {calls['llm_load']}, LLM unload: {calls['llm_unload']}")
    order = []
    for _, job, stage in timeline:
        if stage == "analyzing":
            order.append(job)
    print("порядок старта:", order)
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name}: title={data['title']!r} llm_model={data.get('llm_model')} titles={len(data.get('titles', []))} "
              f"music={data.get('music_file')} subs={list(data.get('subtitle_files', {}))} errors={data.get('llm_errors')}")
    app.quit()


if __name__ == "__main__":
    main()
