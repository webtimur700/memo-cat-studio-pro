"""Готовит входные данные для scripts/benchmark_llm.py: транскрипция и кадр
обложки для нескольких реальных моментов видео.

    python scripts/benchmark_prepare_inputs.py ~/Видео/myvideo.mp4 150 210 990 60
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entities.moment import Moment  # noqa: E402
from core.entities.settings import UserSettings  # noqa: E402
from pipeline.pipeline_runner import PipelineRunner  # noqa: E402
from video.frame_extractor import FrameExtractor  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "llm_benchmark_inputs"
MOMENT_LEN_SEC = 30.0


def main() -> None:
    video = Path(sys.argv[1]).expanduser()
    starts = [float(x) for x in sys.argv[2:]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = PipelineRunner(output_dir=Path("/tmp/bench_tmp"))
    settings = UserSettings()

    inputs = []
    for start in starts:
        moment = Moment(start, start + MOMENT_LEN_SEC, 80, 0.5, ())
        words = runner._transcribe_moment(video, moment, settings)
        with FrameExtractor(video) as extractor:
            _, frame = extractor.best_frame_for_cover(moment.start_sec, moment.end_sec)
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (768, round(768 * height / width)), interpolation=cv2.INTER_AREA)
        image_name = f"moment_{int(start)}.jpg"
        cv2.imwrite(str(OUT_DIR / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        inputs.append({
            "id": f"m{int(start)}",
            "video": video.name,
            "start_sec": start,
            "transcript": " ".join(w.text for w in words),
            "image": image_name,
        })
        print(inputs[-1]["id"], len(words), "слов:", inputs[-1]["transcript"][:100])

    (OUT_DIR / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
