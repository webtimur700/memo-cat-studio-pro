"""Замер времени стадий пайплайна на реальном видео (без разгрузки/загрузки LLM-моделей).

    python scripts/profile_pipeline.py video.mp4 [--llm] [--keep DIR] [--json out.json]

Берёт накопленные секундомеры core/stage_timer.py, печатает таблицу «стадия — секунд всего —
секунд на клип». --llm подключает LM Studio (если она недоступна, пайплайн деградирует
без LLM, а таблица честно покажет ~0 у LLM-стадий).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import stage_timer  # noqa: E402
from core.entities.settings import UserSettings  # noqa: E402
from pipeline.pipeline_runner import PipelineRunner  # noqa: E402


class _FakeLLM:
    """Отвечает канонической заглушкой через `delay` секунд: замер конвейера при известной задержке LLM."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        time.sleep(self.delay)
        if "нумерованным списком" in system_prompt:
            return "\n".join(f"{i}. Заголовок {i}" for i in range(1, 11))
        if "Отвечай СТРОГО списком хештегов" in system_prompt:
            return " ".join(f"#тег{i}" for i in range(1, 31))
        return "Описание клипа."

    def list_models(self) -> list[str]:
        return ["fake"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--keep", help="куда сохранить клипы (по умолчанию — временная папка)")
    parser.add_argument("--fake-llm-sec", type=float, default=0.0,
                        help="вместо LM Studio — заглушка, отвечающая на каждый из 3 запросов за N/3 с (для замера перекрытия LLM и кодирования)")
    parser.add_argument("--no-overlap", action="store_true", help="тексты LLM по очереди с кодированием (как было до оптимизации)")
    parser.add_argument("--separate-llm", action="store_true", help="тексты клипа тремя запросами (как до объединения), для сравнения")
    parser.add_argument("--separate-decode", action="store_true", help="сцены и сканирование читают видео по отдельности (как до общего прохода), для сравнения")
    parser.add_argument("--model", help="ключ модели LM Studio (иначе выбирает селектор по памяти)")
    parser.add_argument("--json")
    parser.add_argument("--encoder", choices=("auto", "vaapi", "x264"), default="auto")
    parser.add_argument("--max-clips", type=int, default=0, help="0 — сколько выберет пайплайн")
    args = parser.parse_args()

    settings = UserSettings()
    settings = replace(settings, viral_score=replace(settings.viral_score, queue_threshold=0))

    settings = replace(settings, export=replace(settings.export, encoder=args.encoder))

    llm = None
    if args.llm:
        from llm.lm_studio_provider import LMStudioConfig
        from llm.managed_provider import ManagedLMStudio

        llm = ManagedLMStudio(LMStudioConfig(base_url="http://localhost:1234/v1", timeout_sec=300, model_override=args.model))
        llm.start_loading_in_background()

    if args.fake_llm_sec > 0:
        llm = _FakeLLM(args.fake_llm_sec / 3)

    def run(out: Path) -> None:
        runner = PipelineRunner(models_dir=ROOT / "models", output_dir=out, llm_provider=llm, overlap_llm=not args.no_overlap,
                                llm_combined=not args.separate_llm, shared_decode=not args.separate_decode)
        stage_timer.reset()
        started = time.perf_counter()
        clips = runner.process_video(Path(args.video).expanduser(), settings)
        total = time.perf_counter() - started
        snap = stage_timer.snapshot()
        n = max(1, len(clips))
        rows = {k: {"sec": round(v[0], 2), "calls": v[1], "sec_per_clip": round(v[0] / n, 2)} for k, v in sorted(snap.items())}
        print(f"\nклипов: {len(clips)}, всего {total:.1f} с, {total / n:.1f} с на клип\n")
        print(f"{'стадия':<52}{'всего, с':>10}{'вызовов':>9}{'на клип, с':>12}")
        for name, row in rows.items():
            print(f"{name:<52}{row['sec']:>10.2f}{row['calls']:>9}{row['sec_per_clip']:>12.2f}")
        if args.json:
            Path(args.json).write_text(json.dumps({"clips": len(clips), "total_sec": round(total, 1), "stages": rows}, ensure_ascii=False, indent=2))
        if hasattr(llm, "release"):
            llm.release()

    if args.keep:
        Path(args.keep).mkdir(parents=True, exist_ok=True)
        run(Path(args.keep))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp))


if __name__ == "__main__":
    main()
