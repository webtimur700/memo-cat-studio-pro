"""Детекция смены сцены через PySceneDetect (Функция 2: "резкая смена сцены").

Использует ContentDetector — детектирует смену по разнице HSV-гистограмм
соседних кадров, что на практике надёжно ловит именно резкие склейки/смену
плана, а не плавные панорамы (которые ContentDetector сознательно игнорирует
при разумном threshold).

ЧЕСТНОЕ ПРИМЕЧАНИЕ: PySceneDetect не установлен в этой песочнице (нет сети
для pip install), поэтому этот файл проверен статическим анализом (py_compile)
и сверен с публичным API PySceneDetect 0.6.x, но не прогнан на реальном видео
в этом окружении. Первый запуск на твоей машине — лучшая проверка; ниже я дам
команду для быстрой самопроверки.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from core.entities.scene_segment import SceneSegment
from core.exceptions import SceneDetectionError

DEFAULT_CONTENT_THRESHOLD = 27.0   # стандартное значение PySceneDetect для "резкой" смены
MIN_SCENE_LEN_FRAMES = 15          # не дробить на сцены короче ~0.5s при 30fps


def _whole_video_scene(duration_sec: float, total_frames: int) -> SceneSegment:
    return SceneSegment(index=0, start_sec=0.0, end_sec=duration_sec, start_frame=0, end_frame=total_frames)


def _segments(scene_list) -> list[SceneSegment]:
    return [
        SceneSegment(
            index=i,
            start_sec=start.get_seconds(),
            end_sec=end.get_seconds(),
            start_frame=start.get_frames(),
            end_frame=end.get_frames(),
        )
        for i, (start, end) in enumerate(scene_list)
    ]


class SceneStream:
    """Тот же ContentDetector и то же уменьшение кадра, что у SceneManager PySceneDetect 0.6.4 (auto_downscale, INTER_LINEAR),
    но кадры приходят снаружи: `feed(номер, кадр BGR)` подряд с нуля, потом `finish()`. Результат совпадает с
    SceneDetector.detect(path) на тех же кадрах (тест tests/unit/test_shared_decode.py)."""

    def __init__(
        self, threshold: float, min_scene_len_frames: int, fps: float, frame_size: tuple[int, int], frame_count: int
    ) -> None:
        from scenedetect import ContentDetector
        from scenedetect.scene_manager import compute_downscale_factor

        self._detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        self._factor = compute_downscale_factor(frame_width=frame_size[0])
        self._fps = fps
        self._frame_count = frame_count
        self._cuts: list[int] = []
        self._frames_fed = 0

    def feed(self, frame_num: int, frame: np.ndarray) -> None:
        if self._factor > 1:
            frame = cv2.resize(
                frame,
                (round(frame.shape[1] / self._factor), round(frame.shape[0] / self._factor)),
                interpolation=cv2.INTER_LINEAR,
            )
        self._cuts += self._detector.process_frame(frame_num, frame)
        self._frames_fed = frame_num + 1

    def finish(self) -> list[SceneSegment]:
        from scenedetect import FrameTimecode
        from scenedetect.scene_manager import get_scenes_from_cuts

        if self._frames_fed == 0:
            return [_whole_video_scene(0.0, 0)]
        self._cuts += self._detector.post_process(self._frames_fed - 1)
        base = FrameTimecode(0, self._fps)
        cut_list = [base + cut for cut in sorted(set(self._cuts))]
        if not cut_list:
            duration = FrameTimecode(self._frame_count, self._fps)
            return [_whole_video_scene(duration.get_seconds(), duration.get_frames())]
        scenes = get_scenes_from_cuts(
            cut_list=cut_list, start_pos=base, end_pos=FrameTimecode(self._frames_fed, self._fps)
        )
        return _segments(scenes)


class SceneDetector:
    def __init__(
        self,
        content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
        min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
    ) -> None:
        self._content_threshold = content_threshold
        self._min_scene_len_frames = min_scene_len_frames

    def detect(self, path: Path) -> list[SceneSegment]:
        try:
            from scenedetect import ContentDetector, SceneManager, open_video
        except ImportError as exc:
            raise SceneDetectionError(
                "Библиотека scenedetect не установлена. Установите: "
                "pip install scenedetect[opencv] --break-system-packages"
            ) from exc

        try:
            video = open_video(str(path))
        except Exception as exc:  # PySceneDetect бросает разные исключения на разные форматы
            raise SceneDetectionError(f"Не удалось открыть видео для детекции сцен: {path}") from exc

        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self._content_threshold,
                min_scene_len=self._min_scene_len_frames,
            )
        )

        try:
            scene_manager.detect_scenes(video=video, show_progress=False)
            raw_scene_list = scene_manager.get_scene_list()
        except Exception as exc:
            raise SceneDetectionError(f"Ошибка детекции сцен для {path}") from exc

        if not raw_scene_list:
            # Если сцены не найдены (например, видео — один непрерывный план),
            # считаем весь ролик одной сценой, а не бросаем ошибку.
            duration = video.duration.get_seconds() if video.duration else 0.0
            total_frames = video.duration.get_frames() if video.duration else 0
            return [_whole_video_scene(duration, total_frames)]

        segments = _segments(raw_scene_list)
        logger.info("Найдено {} сцен(ы) в {}", len(segments), path.name)
        return segments

    def stream(self, fps: float, frame_size: tuple[int, int], frame_count: int) -> "SceneStream":
        """Детекция сцен по кадрам, которые подаёт кто-то другой (общий проход декодера, video/shared_decode.py)."""
        return SceneStream(self._content_threshold, self._min_scene_len_frames, fps, frame_size, frame_count)

    def nearest_scene_boundary(self, segments: list[SceneSegment], timestamp_sec: float) -> float:
        """Находит ближайшую границу сцены к заданной временной метке —
        используется cutting/boundary_refiner.py, чтобы не резать посреди сцены.
        """
        if not segments:
            return timestamp_sec

        boundaries = sorted(
            {s.start_sec for s in segments} | {s.end_sec for s in segments}
        )
        return min(boundaries, key=lambda b: abs(b - timestamp_sec))
