"""Оркестратор полного пайплайна обработки одного видео — связывает все
движки (Шаги 5-10) в одну сквозную операцию: анализ -> скоринг -> нарезка ->
кроп -> субтитры -> брендинг -> экспорт.

ЧЕСТНАЯ ДЕГРАДАЦИЯ — ключевой принцип модуля. ML-модели (YOLO11 ONNX,
faster-whisper) могут быть не скачаны, LM Studio может быть не запущена.
Пайплайн в этих случаях не падает и не "молчит" — он деградирует до более
простого, но реального результата:
  - нет YOLO-модели -> скоринг идёт по motion-эвристике (разница соседних
    кадров через OpenCV, без всякого ML), автокадрирование — статичный
    центральный кроп вместо слежения за объектом
  - нет faster-whisper -> экспорт без субтитров
  - LM Studio недоступна -> заголовок собирается детерминированно из
    метаданных момента, без LLM
Каждая деградация логируется через loguru — видно, чего именно не хватает
на конкретной машине, а не просто "тихо хуже работает".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.entities.clip import Clip
from core.entities.detection import Detection
from core.entities.moment import Moment
from core.entities.scene_segment import SceneSegment
from core.entities.settings import UserSettings, ViralScoreSettings
from core.exceptions import MemoCatError
from core.entities.subtitle import WordTiming
from cutting.clip_selector_service import WindowScore, select_moments
from effects.collision_detector import BannerPosition, resolve_banner_position
from export.dynamic_crop import CropSample
from export.export_service import ExportPlan, ExportService
from llm.content_generator import ClipContent, generate_clip_content
from scoring.viral_score_service import ScoreInputs, compute_viral_score
from subtitles.ass_renderer import render_ass
from subtitles.subtitle_service import group_words_into_segments
from video.ffmpeg_wrapper import FFmpegWrapper
from video.frame_extractor import FrameExtractor
from video.ingestion_service import IngestionService
from video.scene_detector import SceneDetector
from vision.smart_crop import CropWindow, SmartCropPlanner
from vision.tracker import ObjectTracker

WINDOW_SEC = 5.0
SAMPLE_FPS_SCAN = 1.0     # частота сэмплирования при поиске моментов
SAMPLE_FPS_CROP = 2.0     # частота сэмплирования при кадрировании внутри момента
MOTION_NORMALIZATION = 25.0  # эмпирический делитель для перевода средней разницы яркости в 0..1

ProgressCallback = Callable[[str, dict], None]


class PipelineError(MemoCatError):
    pass


def _frame_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Motion-эвристика без ML: средняя абсолютная разница яркости соседних
    кадров, нормализованная в 0..1. Работает всегда, даже без единой
    скачанной модели.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    return min(1.0, float(diff.mean()) / MOTION_NORMALIZATION)


@dataclass
class _DetectorHandle:
    """Скрывает отсутствие YOLO-модели за единым интерфейсом — остальной код
    пайплайна не ветвится на "есть модель / нет модели" в десяти местах.
    """

    detector: object | None

    @property
    def available(self) -> bool:
        return self.detector is not None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.detector is None:
            return []
        return self.detector.detect(frame)  # type: ignore[attr-defined]


def _try_load_yolo(models_dir: Path) -> _DetectorHandle:
    model_path = models_dir / "yolo11n.onnx"
    if not model_path.exists():
        logger.warning(
            "YOLO11 модель не найдена ({}) — скоринг и автокадрирование "
            "работают в упрощённом режиме (motion-эвристика, без детекции "
            "животных). Запустите scripts/download_models.py для полного AI-анализа.",
            model_path,
        )
        return _DetectorHandle(detector=None)

    try:
        from vision.yolo_detector import YoloDetector

        return _DetectorHandle(detector=YoloDetector(model_path))
    except Exception as exc:
        logger.warning("Не удалось загрузить YOLO11 ({}): {} — работаю без детекции", model_path, exc)
        return _DetectorHandle(detector=None)


class PipelineRunner:
    def __init__(
        self,
        models_dir: Path | None = None,
        output_dir: Path | None = None,
        llm_provider: object | None = None,
    ) -> None:
        self._models_dir = models_dir or Path("models")
        self._output_dir = output_dir or Path("export/output")
        self._llm_provider = llm_provider
        self._transcriber = None  # создаётся на каждый process_video
        self._llm_unavailable = False  # после первой недоступности LLM не долбимся в неё до конца видео

    def process_video(
        self,
        video_path: Path,
        settings: UserSettings,
        progress: ProgressCallback | None = None,
    ) -> list[Clip]:
        def emit(stage: str, **data: object) -> None:
            if progress is not None:
                progress(stage, data)

        self._transcriber = None
        self._llm_unavailable = False
        emit("analyzing")
        source = IngestionService().ingest(video_path)
        detector = _try_load_yolo(self._models_dir)
        scenes = self._detect_scenes_safely(video_path)

        emit("scoring")
        window_scores = self._scan_windows(video_path, source.duration_sec, detector, scenes)

        emit("cutting")
        moments = select_moments(window_scores, settings, source.duration_sec)
        emit("cutting", moments_found=len(moments))

        if not moments:
            logger.info("Для {} не найдено моментов выше порога Viral Score", video_path.name)
            return []

        self._output_dir.mkdir(parents=True, exist_ok=True)

        clips: list[Clip] = []
        for index, moment in enumerate(moments):
            emit("reframing", moment_index=index, total=len(moments))
            clip = self._export_moment(video_path, source, moment, index, detector, settings)
            if clip is not None:
                clips.append(clip)

        emit("done", moments_found=len(clips))
        return clips

    # ------------------------------------------------------------------
    def _detect_scenes_safely(self, video_path: Path) -> list[SceneSegment]:
        try:
            return SceneDetector().detect(video_path)
        except Exception as exc:
            logger.warning("Детекция сцен недоступна для {}: {} — считаю 0 смен сцены", video_path.name, exc)
            return []

    def _scan_windows(
        self,
        video_path: Path,
        duration_sec: float,
        detector: _DetectorHandle,
        scenes: list[SceneSegment],
    ) -> list[WindowScore]:
        results: list[WindowScore] = []
        default_weights = ViralScoreSettings()

        with FrameExtractor(video_path) as extractor:
            window_start = 0.0
            while window_start < duration_sec:
                window_end = min(window_start + WINDOW_SEC, duration_sec)

                prev_gray: np.ndarray | None = None
                motion_samples: list[float] = []
                detections_in_window: list[Detection] = []
                frames_with_detection = 0
                total_frames = 0

                for _timestamp, frame in extractor.frames_in_range(
                    window_start, window_end, sample_fps=SAMPLE_FPS_SCAN
                ):
                    total_frames += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        motion_samples.append(_frame_motion_score(prev_gray, gray))
                    prev_gray = gray

                    if detector.available:
                        frame_detections = detector.detect(frame)
                        if frame_detections:
                            frames_with_detection += 1
                            detections_in_window.extend(frame_detections)

                motion_intensity = sum(motion_samples) / len(motion_samples) if motion_samples else 0.0
                # Честный fallback: без YOLO детекции "присутствия животного" не
                # существует, поэтому переиспользуем motion_intensity как
                # ближайший доступный сигнал — иначе вес face_prominence просто
                # обнулял бы часть формулы для всех, у кого нет скачанной модели.
                detection_presence = (
                    frames_with_detection / total_frames
                    if detector.available and total_frames > 0
                    else motion_intensity
                )
                scene_change_count = sum(1 for s in scenes if window_start < s.start_sec < window_end)

                score = compute_viral_score(
                    ScoreInputs(motion_intensity, detection_presence, scene_change_count),
                    default_weights,
                )

                results.append(
                    WindowScore(
                        start_sec=window_start,
                        end_sec=window_end,
                        viral_score=score,
                        motion_intensity=motion_intensity,
                        detections=tuple(detections_in_window[:5]),
                    )
                )
                window_start = window_end

        return results

    def _export_moment(
        self,
        video_path: Path,
        source,
        moment: Moment,
        index: int,
        detector: _DetectorHandle,
        settings: UserSettings,
    ) -> Clip | None:
        crop_samples = self._build_crop_samples(video_path, source, moment, detector, settings)
        words = self._transcribe_moment(video_path, moment, settings)
        subtitle_ass_path = self._write_subtitles(video_path, moment, words, settings)
        banner_rect = self._build_banner_rect(source, settings)

        output_path = self._output_dir / f"{video_path.stem}_moment{index + 1}.mp4"

        plan = ExportPlan(
            source_path=video_path,
            output_path=output_path,
            crop_samples=crop_samples,
            subtitle_ass_path=subtitle_ass_path,
            banner_rect=banner_rect,
            banner_text_lines=list(settings.branding.banner_text_lines),
            banner_appear_at_sec=4.0,
            banner_duration_sec=5.0,
            settings=settings.export,
            source_start_sec=moment.start_sec,
        )

        try:
            ExportService().export_clip(plan)
        except Exception as exc:
            logger.error("Экспорт момента {} из {} не удался: {}", index + 1, video_path.name, exc)
            return None
        finally:
            if subtitle_ass_path is not None:
                subtitle_ass_path.unlink(missing_ok=True)

        transcript = " ".join(w.text for w in words)
        content = self._generate_content(transcript)
        title = content.titles[0] if content.titles else self._default_title(moment)
        metadata_path = self._write_metadata(output_path, video_path, moment, transcript, title, content)

        return Clip(
            moment=moment,
            output_path=output_path,
            title=title,
            description=content.description,
            hashtags=content.hashtags,
            titles=content.titles,
            transcript=transcript,
            metadata_path=metadata_path,
        )

    # ------------------------------------------------------------------
    def _generate_content(self, transcript: str) -> ClipContent:
        """Заголовки/описание/хештеги от LLM по тексту транскрипции момента. Без
        LLM-провайдера или если LM Studio недоступна — пустой результат (клип
        получает заголовок по умолчанию), пайплайн не падает.
        """
        if self._llm_provider is None or self._llm_unavailable:
            return ClipContent()
        content = generate_clip_content(self._llm_provider, transcript)
        if content.is_empty and content.errors:
            self._llm_unavailable = True
        return content

    @staticmethod
    def _default_title(moment: Moment) -> str:
        return f"Момент {moment.start_sec:.0f}s (score {moment.viral_score})"

    def _write_metadata(
        self, output_path: Path, video_path: Path, moment: Moment, transcript: str, title: str, content: ClipContent
    ) -> Path | None:
        metadata_path = output_path.with_suffix(".json")
        payload = {
            "source_video": video_path.name,
            "clip_file": output_path.name,
            "start_sec": moment.start_sec,
            "end_sec": moment.end_sec,
            "viral_score": moment.viral_score,
            "title": title,
            "titles": list(content.titles),
            "description": content.description,
            "hashtags": list(content.hashtags),
            "transcript": transcript,
            "llm_errors": list(content.errors),
        }
        try:
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось записать {}: {}", metadata_path.name, exc)
            return None
        return metadata_path

    def _get_transcriber(self, settings: UserSettings):
        """Один WhisperTranscriber на весь прогон видео: модель грузится в память
        один раз (ленивo, при первой транскрипции), а не заново для каждого момента.
        """
        if self._transcriber is None:
            from subtitles.subtitle_service import WhisperTranscriber

            self._transcriber = WhisperTranscriber(
                model_size=settings.subtitles.model_size, compute_type=settings.subtitles.compute_type
            )
        return self._transcriber

    def _transcribe_moment(self, video_path: Path, moment: Moment, settings: UserSettings) -> list[WordTiming]:
        """Слова момента с таймингами ОТНОСИТЕЛЬНО начала момента. Транскрибируется
        только звук самого момента (15-60 с), а не всего видео. При любой
        проблеме возвращает [] и пишет warning — пайплайн идёт дальше без субтитров.
        """
        audio_path = self._output_dir / f"_tmp_audio_{video_path.stem}_{int(moment.start_sec)}.wav"
        try:
            FFmpegWrapper().extract_audio_track(
                video_path, audio_path, start_sec=moment.start_sec, duration_sec=moment.duration_sec
            )
            return self._get_transcriber(settings).transcribe(audio_path)
        except Exception as exc:
            logger.warning(
                "Транскрипция недоступна для момента {:.1f}s видео {}: {} — без субтитров и без текста для LLM",
                moment.start_sec, video_path.name, exc,
            )
            return []
        finally:
            audio_path.unlink(missing_ok=True)

    def _write_subtitles(
        self, video_path: Path, moment: Moment, words: list[WordTiming], settings: UserSettings
    ) -> Path | None:
        if not words or not settings.subtitles.burn_in:
            return None
        ass_path = self._output_dir / f"_tmp_subs_{video_path.stem}_{int(moment.start_sec)}.ass"
        try:
            segments = group_words_into_segments(words)
            ass_content = render_ass(segments, style_preset=settings.subtitles.style_preset)
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            ass_path.write_text(ass_content, encoding="utf-8")
            return ass_path
        except Exception as exc:
            logger.warning("Не удалось собрать субтитры для момента {:.1f}s: {}", moment.start_sec, exc)
            return None

    def _build_banner_rect(self, source, settings: UserSettings) -> tuple[int, int, int, int]:
        width, height = settings.export.width, settings.export.height
        banner_width, banner_height = int(width * 0.85), 220

        try:
            preferred_position = BannerPosition(settings.branding.banner_position)
        except ValueError:
            preferred_position = BannerPosition.BOTTOM_CENTER

        # animal_head_region=None: полноценный collision detection требует
        # покадровой позиции животного на протяжении всего показа плашки —
        # в этой версии оркестратора банер ставится статично по настройкам
        # пользователя, без динамического обхода коллизий (это отдельный
        # кусок доработки поверх effects/collision_detector.py).
        resolution = resolve_banner_position(preferred_position, None, width, height, banner_width, banner_height)
        rect = resolution.final_rect
        return (int(rect.x1), int(rect.y1), int(rect.x2), int(rect.y2))
