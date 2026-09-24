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

from audio.music_mixer import list_tracks, mix_music, pick_track, probe_duration, speech_intervals
from core.entities.clip import Clip
from core.entities.detection import BoundingBox, Detection, is_animal_class
from core.entities.moment import Moment
from core.entities.scene_segment import SceneSegment
from core.entities.settings import UserSettings, ViralScoreSettings
from core.exceptions import MemoCatError
from core.entities.subtitle import WordTiming
from cutting.clip_selector_service import WindowScore, select_moments
from effects.branding_overlay import BrandingOverlay, resolve_logo_path
from effects.safe_zone import SafeZone
from effects.cover_generator import generate_cover
from effects.collision_detector import BannerPosition, resolve_banner_position_over_time
from export.dynamic_crop import CropSample
from export.export_service import ExportPlan, ExportService
from llm.content_generator import ClipContent, generate_clip_content
from pipeline.shared_models import SharedModels
from scoring.viral_score_service import ScoreInputs, compute_viral_score
from subtitles.ass_renderer import estimate_subtitle_band_height, render_ass, render_srt
from subtitles.subtitle_service import group_words_into_segments
from video.ffmpeg_wrapper import FFmpegWrapper
from video.frame_extractor import FrameExtractor
from video.ingestion_service import IngestionService
from video.scene_detector import SceneDetector
from vision.smart_crop import CropWindow, SmartCropPlanner
from vision.mediapipe_face import AnimalHeadRegionEstimator
from vision.tracker import ObjectTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOW_SEC = 5.0
MIN_PAUSE_SEC = 0.35          # промежуток между словами, считающийся паузой (можно резать)
SPEECH_EDGE_MARGIN_SEC = 0.15
MIN_RELIABLE_WORDS = 3          # меньше слов при неуверенном определении языка — считаем шумом, а не речью
RELIABLE_LANGUAGE_PROB = 0.8
SUBTITLE_BANNER_GAP_PX = 24   # зазор между плашкой и полосой субтитров
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


def frame_to_jpeg(frame: np.ndarray, max_side: int = 768, quality: int = 85) -> bytes | None:
    """Кадр (BGR) -> JPEG для vision-модели: уменьшенный, чтобы не раздувать промпт."""
    height, width = frame.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else None


def _motion_direction_x(detections: list[Detection]) -> float:
    """-1..1: горизонтальный сдвиг центра bbox между двумя последними детекциями
    трека относительно ширины bbox (голова животного смещена по ходу движения)."""
    if len(detections) < 2:
        return 0.0
    prev, curr = detections[-2].bbox, detections[-1].bbox
    prev_cx, curr_cx = (prev.x1 + prev.x2) / 2, (curr.x1 + curr.x2) / 2
    return max(-1.0, min(1.0, (curr_cx - prev_cx) / max(curr.width, 1.0)))


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


class PipelineRunner:
    def __init__(
        self,
        models_dir: Path | None = None,
        output_dir: Path | None = None,
        llm_provider: object | None = None,
        assets_dir: Path | None = None,
        shared_models: SharedModels | None = None,
    ) -> None:
        # assets/logo/logo.png (если положили) подхватывается вместо логотипа по умолчанию
        self._assets_dir = assets_dir or PROJECT_ROOT / "assets"
        self._models_dir = models_dir or Path("models")
        self._output_dir = output_dir or Path("export/output")
        self._llm_provider = llm_provider
        # YOLO и Whisper: если раннер создан очередью — модели общие на все видео, иначе свои на этот раннер
        self._shared = shared_models or SharedModels(self._models_dir)
        self._llm_unavailable = False  # после первой недоступности LLM не долбимся в неё до конца видео
        # транскрипции диапазонов видео (слова с АБСОЛЮТНЫМИ таймкодами): и поиск пауз для
        # границ моментов, и субтитры берут слова отсюда, а не транскрибируют звук дважды
        self._word_cache: list[tuple[float, float, list[WordTiming], str | None]] = []
        self.last_moment_language: str | None = None   # язык речи в последнем транскрибированном моменте

    def process_video(
        self,
        video_path: Path,
        settings: UserSettings,
        progress: ProgressCallback | None = None,
    ) -> list[Clip]:
        def emit(stage: str, **data: object) -> None:
            if progress is not None:
                progress(stage, data)

        self._llm_unavailable = False
        self._word_cache = []
        emit("analyzing")
        source = IngestionService().ingest(video_path)
        detector = _DetectorHandle(detector=self._shared.detector())
        scenes = self._detect_scenes_safely(video_path)

        emit("scoring")
        window_scores = self._scan_windows(video_path, source.duration_sec, detector, scenes)

        emit("cutting")
        scene_boundaries = sorted(
            {s.start_sec for s in scenes if s.start_sec > 0} | {s.end_sec for s in scenes if s.end_sec < source.duration_sec}
        )
        moments = select_moments(
            window_scores,
            settings,
            source.duration_sec,
            scene_boundaries=scene_boundaries,
            pause_finder=lambda lo, hi: self._find_speech_pauses(video_path, lo, hi, settings),
        )
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
                if window_end - window_start < 1.0 and results:
                    break  # хвост короче секунды (например 180.0-180.02) — не окно

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
                        # присутствие в кадре считаем только по животным: человек не делает момент "вирусным"
                        frame_detections = [d for d in detector.detect(frame) if is_animal_class(d.class_id)]
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
        crop_samples, head_regions = self._build_crop_samples(video_path, source, moment, detector, settings)
        words = self._transcribe_moment(video_path, moment, settings)
        original_transcript = " ".join(w.text for w in words)
        speech_words = words   # тайминги речи для приглушения музыки (перевод их не меняет, но берём оригинал)
        words = self._translate_subtitles(words, self.last_moment_language, settings)
        zone = SafeZone.from_settings((settings.export.width, settings.export.height), settings.safe_zone)
        output_path = self._output_dir / f"{video_path.stem}_moment{index + 1}.mp4"
        subtitle_ass_path, subtitle_files = self._write_subtitles(output_path, words, settings, zone)
        branding = self._build_branding(settings, zone)
        banner_appear_sec, banner_duration_sec = 4.0, 5.0
        subtitles_during_banner = subtitle_ass_path is not None and any(
            w.end_sec > banner_appear_sec and w.start_sec < banner_appear_sec + banner_duration_sec for w in words
        )
        banner_rect = self._build_banner_rect(
            settings, crop_samples, head_regions, branding, banner_appear_sec, banner_duration_sec,
            zone=zone, subtitles_during_banner=subtitles_during_banner,
        )

        plan = ExportPlan(
            source_path=video_path,
            output_path=output_path,
            crop_samples=crop_samples,
            subtitle_ass_path=subtitle_ass_path,
            banner_rect=banner_rect,
            banner_text_lines=list(settings.branding.banner_text_lines),
            banner_appear_at_sec=banner_appear_sec,
            banner_duration_sec=banner_duration_sec,
            branding=branding,
            settings=settings.export,
            source_start_sec=moment.start_sec,
        )

        try:
            ExportService().export_clip(plan)
        except Exception as exc:
            logger.error("Экспорт момента {} из {} не удался: {}", index + 1, video_path.name, exc)
            for path in subtitle_files.values():   # клипа нет — файлы субтитров без него не нужны
                path.unlink(missing_ok=True)
            return None
        finally:
            if subtitle_ass_path is not None and subtitle_ass_path not in subtitle_files.values():
                subtitle_ass_path.unlink(missing_ok=True)

        music_file = self._add_music(output_path, speech_words, settings)
        transcript = " ".join(w.text for w in words)
        cover_frame = self._pick_cover_frame(video_path, moment)
        content = self._generate_content(transcript, cover_frame)
        title = content.titles[0] if content.titles else self._default_title(moment)
        cover_path = self._generate_cover(cover_frame, moment, crop_samples, title, output_path, settings)
        metadata_path = self._write_metadata(
            output_path, video_path, moment, transcript, title, content, cover_path,
            transcript_original=original_transcript, language=self.last_moment_language,
            subtitle_files=subtitle_files,
            music_file=music_file,
        )

        return Clip(
            moment=moment,
            output_path=output_path,
            title=title,
            description=content.description,
            hashtags=content.hashtags,
            titles=content.titles,
            transcript=transcript,
            metadata_path=metadata_path,
            cover_path=cover_path,
            subtitle_paths=tuple(subtitle_files.values()),
        )

    # ------------------------------------------------------------------
    def _add_music(self, output_path: Path, speech_words: list[WordTiming], settings: UserSettings) -> str | None:
        """Фоновая музыка из assets/music/ (приглушается на речи). Возвращает имя трека или None (без музыки)."""
        audio = settings.audio
        if not audio.music_enabled:
            return None
        library = Path(audio.music_library_path)
        library = library if library.is_absolute() else PROJECT_ROOT / library
        tracks = list_tracks(library)
        if not tracks:
            logger.info("Музыка не добавлена к {}: в {} нет треков (положите туда свою музыку без авторских ограничений)",
                        output_path.name, library)
            return None
        try:
            duration = probe_duration(output_path)
            track = pick_track(tracks, output_path.stem, duration)
            intervals = speech_intervals(speech_words, duration) if audio.duck_on_speech else []
            mix_music(output_path, track, duration, audio.music_volume, intervals, audio.duck_level_db,
                      audio_codec=settings.export.codec_audio)
            return track.name
        except Exception as exc:
            logger.warning("Не удалось добавить музыку к {}: {} — клип остаётся с оригинальным звуком", output_path.name, exc)
            return None

    def _build_branding(self, settings: UserSettings, zone: SafeZone | None = None) -> BrandingOverlay | None:
        """Логотип (assets/logo/logo.png или "Memo Cat" по умолчанию) + Subscribe."""
        try:
            branding = BrandingOverlay.build(
                frame_size=(settings.export.width, settings.export.height),
                logo_path=resolve_logo_path(self._assets_dir),
                logo_position=settings.branding.logo_position,
                subscribe_enabled=settings.branding.subscribe_button_enabled,
                zone=zone,
            )
        except Exception as exc:
            logger.warning("Брендинг (логотип/Subscribe) недоступен: {} — экспорт без него", exc)
            return None
        return None if branding.is_empty else branding

    def _generate_content(self, transcript: str, cover_frame: tuple[float, np.ndarray] | None = None) -> ClipContent:
        """Заголовки/описание/хештеги от LLM по тексту транскрипции момента (и по кадру обложки,
        если модель умеет vision). Без LLM-провайдера или если LM Studio недоступна — пустой
        результат (клип получает заголовок по умолчанию), пайплайн не падает.
        """
        if self._llm_provider is None or self._llm_unavailable:
            return ClipContent()
        image_jpeg = None
        if cover_frame is not None and getattr(self._llm_provider, "supports_vision", False):
            image_jpeg = frame_to_jpeg(cover_frame[1])
        content = generate_clip_content(self._llm_provider, transcript, image_jpeg=image_jpeg)
        if content.is_empty and content.errors:
            self._llm_unavailable = True
        return content

    def _pick_cover_frame(self, video_path: Path, moment: Moment) -> tuple[float, np.ndarray] | None:
        """Самый резкий кадр момента (нужен и LLM как картинка, и обложке)."""
        try:
            with FrameExtractor(video_path) as extractor:
                return extractor.best_frame_for_cover(moment.start_sec, moment.end_sec)
        except Exception as exc:
            logger.warning("Кадр обложки для момента {:.1f}s не получен: {}", moment.start_sec, exc)
            return None

    def _llm_model_name(self) -> str | None:
        selection = getattr(self._llm_provider, "selection", None)
        return selection.model_key if selection is not None else getattr(self._llm_provider, "last_model", None) or None

    @staticmethod
    def _default_title(moment: Moment) -> str:
        return f"Момент {moment.start_sec:.0f}s (score {moment.viral_score})"

    def _generate_cover(
        self,
        cover_frame: tuple[float, np.ndarray] | None,
        moment: Moment,
        crop_samples: list[CropSample],
        title: str,
        output_path: Path,
        settings: UserSettings,
    ) -> Path | None:
        """<имя_клипа>_cover.png: самый резкий кадр момента, обрезанный тем же окном
        автокадрирования, что и в клипе (вертикаль 9:16), + заголовок.
        Любой сбой — warning и клип без обложки, пайплайн не падает.
        """
        if cover_frame is None:
            return None
        cover_path = output_path.with_name(f"{output_path.stem}_cover.png")
        try:
            timestamp, frame = cover_frame
            relative_t = timestamp - moment.start_sec
            nearest = min(crop_samples, key=lambda cs: abs(cs.timestamp_sec - relative_t))
            height, width = frame.shape[:2]
            window = nearest.window
            x1, x2 = sorted((max(0, int(window.x1)), min(width, int(window.x2))))
            y1, y2 = sorted((max(0, int(window.y1)), min(height, int(window.y2))))
            if x2 - x1 > 1 and y2 - y1 > 1:
                frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(
                frame, (settings.export.width, settings.export.height), interpolation=cv2.INTER_CUBIC
            )

            cover = generate_cover(frame, title)
            cover.convert("RGB").save(cover_path, format="PNG")
            return cover_path
        except Exception as exc:
            logger.warning("Обложка для {} не создана: {}", output_path.name, exc)
            return None

    def _write_metadata(
        self,
        output_path: Path,
        video_path: Path,
        moment: Moment,
        transcript: str,
        title: str,
        content: ClipContent,
        cover_path: Path | None = None,
        transcript_original: str = "",
        language: str | None = None,
        subtitle_files: dict[str, Path] | None = None,
        music_file: str | None = None,
    ) -> Path | None:
        metadata_path = output_path.with_suffix(".json")
        payload = {
            "source_video": video_path.name,
            "clip_file": output_path.name,
            "cover_file": cover_path.name if cover_path else None,
            "start_sec": moment.start_sec,
            "end_sec": moment.end_sec,
            "viral_score": moment.viral_score,
            "title": title,
            "titles": list(content.titles),
            "description": content.description,
            "hashtags": list(content.hashtags),
            "transcript": transcript,
            "transcript_original": transcript_original if transcript_original != transcript else "",
            "speech_language": language,
            "music_file": music_file,
            "subtitle_files": {fmt: path.name for fmt, path in (subtitle_files or {}).items()},
            "llm_errors": list(content.errors),
            "llm_model": self._llm_model_name(),
            "vision_used": content.used_image,
        }
        try:
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось записать {}: {}", metadata_path.name, exc)
            return None
        return metadata_path

    def _build_crop_samples(
        self, video_path: Path, source, moment: Moment, detector: _DetectorHandle, settings: UserSettings
    ) -> tuple[list[CropSample], list[BoundingBox | None]]:
        """Сэмплы автокадрирования + для каждого сэмпла зона головы животного в
        координатах ИСХОДНОГО кадра (None — животного нет / детектор недоступен).
        """
        planner = SmartCropPlanner(
            source.width,
            source.height,
            smoothing_alpha=settings.reframe.tracking_smoothing_alpha,
            min_subject_area_ratio=settings.reframe.min_subject_area_ratio,
            max_zoom_factor=settings.reframe.max_zoom_factor,
        )
        tracker = ObjectTracker()
        samples: list[CropSample] = []
        head_regions: list[BoundingBox | None] = []
        head_estimator = AnimalHeadRegionEstimator()

        with FrameExtractor(video_path) as extractor:
            for timestamp, frame in extractor.frames_in_range(
                moment.start_sec, moment.end_sec, sample_fps=SAMPLE_FPS_CROP
            ):
                relative_t = timestamp - moment.start_sec
                detection = None
                head_region: BoundingBox | None = None

                if detector.available:
                    frame_detections = detector.detect(frame)
                    retimed = [
                        Detection(d.class_id, d.class_name, d.confidence, d.bbox, relative_t)
                        for d in frame_detections
                    ]
                    tracker.update(retimed)
                    primary = tracker.primary_track()
                    if primary is not None:
                        detection = primary.last_detection
                        if primary.is_active:
                            head_region = head_estimator.estimate(detection, _motion_direction_x(primary.detections))

                window = planner.plan_frame(detection)
                head_regions.append(head_region)
                samples.append(
                    CropSample(
                        relative_t,
                        CropWindow(window.x1, window.y1, window.x2, window.y2, window.zoom_factor, relative_t),
                    )
                )

        if not samples:
            # Гарантированный fallback, если не удалось прочитать ни одного
            # кадра момента — статичный центральный кроп на весь момент.
            center = SmartCropPlanner(source.width, source.height).plan_frame(None)
            samples = [
                CropSample(0.0, CropWindow(center.x1, center.y1, center.x2, center.y2, 1.0, 0.0)),
                CropSample(
                    moment.duration_sec,
                    CropWindow(center.x1, center.y1, center.x2, center.y2, 1.0, moment.duration_sec),
                ),
            ]
            head_regions = [None, None]
        elif samples[-1].timestamp_sec < moment.duration_sec - 1e-3:
            # Сэмплы идут с шагом 1/SAMPLE_FPS_CROP, последний обычно не доходит до
            # конца момента; длина клипа берётся из последнего сэмпла, поэтому
            # без замыкающей точки клип получался бы короче момента.
            last = samples[-1].window
            samples.append(
                CropSample(
                    moment.duration_sec,
                    CropWindow(last.x1, last.y1, last.x2, last.y2, last.zoom_factor, moment.duration_sec),
                )
            )
            head_regions.append(head_regions[-1])

        return samples, head_regions

    def _get_transcriber(self, settings: UserSettings):
        """Whisper общий на все видео очереди: модель грузится в память один раз (лениво)."""
        return self._shared.transcriber(settings.subtitles.model_size, settings.subtitles.compute_type)

    def _transcribe_range(self, video_path: Path, start: float, end: float, settings: UserSettings) -> list[WordTiming]:
        """Слова диапазона [start, end] с АБСОЛЮТНЫМИ таймкодами. Транскрибируется
        только звук диапазона; результат кэшируется на время обработки видео. При
        любой проблеме возвращает [] и пишет warning — пайплайн идёт дальше.
        """
        for cache_start, cache_end, cached, cached_language in self._word_cache:
            if cache_start <= start + 0.01 and cache_end >= end - 0.01:
                self.last_moment_language = cached_language
                return [w for w in cached if w.end_sec > start and w.start_sec < end]

        audio_path = self._output_dir / f"_tmp_audio_{video_path.stem}_{int(start * 1000)}.wav"
        try:
            FFmpegWrapper().extract_audio_track(video_path, audio_path, start_sec=start, duration_sec=end - start)
            transcriber = self._get_transcriber(settings)
            with self._shared.transcribe_lock:   # last_language — состояние общего транскрайбера
                relative = transcriber.transcribe(audio_path, language=settings.subtitles.language)
                language = getattr(transcriber, "last_language", None)
                probability = getattr(transcriber, "last_language_probability", 1.0)
            if 0 < len(relative) < MIN_RELIABLE_WORDS and probability < RELIABLE_LANGUAGE_PROB:
                # 1-2 слова и язык определён неуверенно (на смехе/шуме Whisper "слышит" корейский и т.п.)
                logger.info(
                    "Расшифровка {:.1f}-{:.1f}s отброшена как шум: {} слов, язык '{}' ({:.0%})",
                    start, end, len(relative), language, probability,
                )
                relative = []
        except Exception as exc:
            logger.warning(
                "Транскрипция недоступна для {:.1f}-{:.1f}s видео {}: {} — без субтитров и без текста для LLM",
                start, end, video_path.name, exc,
            )
            return []
        finally:
            audio_path.unlink(missing_ok=True)

        absolute = [WordTiming(w.text, w.start_sec + start, w.end_sec + start) for w in relative]
        self._word_cache.append((start, end, absolute, language))
        self.last_moment_language = language
        return absolute

    def _transcribe_moment(self, video_path: Path, moment: Moment, settings: UserSettings) -> list[WordTiming]:
        """Слова момента с таймингами ОТНОСИТЕЛЬНО начала момента. Транскрибируется
        только звук самого момента (или берётся из кэша уже транскрибированного диапазона)."""
        words = self._transcribe_range(video_path, moment.start_sec, moment.end_sec, settings)
        return [
            WordTiming(w.text, max(0.0, w.start_sec - moment.start_sec), min(moment.duration_sec, w.end_sec - moment.start_sec))
            for w in words
            if moment.start_sec - 0.01 <= w.start_sec < moment.end_sec
        ]

    def _find_speech_pauses(self, video_path: Path, start: float, end: float, settings: UserSettings) -> list[float]:
        """Моменты пауз речи в диапазоне: середины промежутков между словами
        (>= MIN_PAUSE_SEC) и края речи. Граница клипа в такой точке не режет фразу."""
        words = self._transcribe_range(video_path, start, end, settings)
        if not words:
            return []
        pauses = [
            (prev.end_sec + nxt.start_sec) / 2
            for prev, nxt in zip(words, words[1:])
            if nxt.start_sec - prev.end_sec >= MIN_PAUSE_SEC
        ]
        pauses.append(max(start, words[0].start_sec - SPEECH_EDGE_MARGIN_SEC))
        pauses.append(min(end, words[-1].end_sec + SPEECH_EDGE_MARGIN_SEC))
        return pauses

    def _translate_subtitles(
        self, words: list[WordTiming], language: str | None, settings: UserSettings
    ) -> list[WordTiming]:
        """Перевод слов на settings.subtitles.translate_to через LLM, если язык речи другой.
        Нет LLM / перевод не удался — остаётся язык оригинала (warning в логе)."""
        target = settings.subtitles.translate_to
        if not words or not target or not language or language == target:
            return words
        if self._llm_provider is None or self._llm_unavailable:
            logger.warning(
                "Речь на языке '{}', а LLM недоступна для перевода на '{}' — субтитры на языке оригинала", language, target
            )
            return words
        from subtitles.translator import translate_words

        translated = translate_words(self._llm_provider, words, target)
        return translated if translated else words

    def _write_subtitles(
        self,
        output_path: Path,
        words: list[WordTiming],
        settings: UserSettings,
        zone: SafeZone | None = None,
    ) -> tuple[Path | None, dict[str, Path]]:
        """Субтитры клипа. Слова уже с таймингами от НАЧАЛА КЛИПА.

        Возвращает (ASS для вжигания в видео или None, {формат: файл рядом с клипом}). SRT и ASS
        (settings.subtitles.export_formats) лежат в папке экспорта под именем клипа — например,
        чтобы отдельно загрузить субтитры на YouTube. Если вжигание включено, а ASS не в списке
        форматов, для ffmpeg делается временный файл (его удаляет вызывающий).
        """
        if not words:
            return None, {}
        formats = settings.subtitles.export_formats
        want_burn = settings.subtitles.burn_in
        saved: dict[str, Path] = {}
        burn_path: Path | None = None
        try:
            segments = group_words_into_segments(words)
            if "srt" in formats:
                saved["srt"] = output_path.with_suffix(".srt")
                saved["srt"].write_text(render_srt(segments), encoding="utf-8")
            if "ass" in formats or want_burn:
                frame_size = (settings.export.width, settings.export.height)
                margin_l, margin_r, margin_v = (
                    zone or SafeZone.from_settings(frame_size, settings.safe_zone)
                ).subtitle_margins(frame_size)
                ass_content = render_ass(
                    segments, style_preset=settings.subtitles.style_preset,
                    play_res_x=frame_size[0], play_res_y=frame_size[1],
                    margin_l=margin_l, margin_r=margin_r, margin_v=margin_v,
                )
                ass_path = output_path.with_suffix(".ass") if "ass" in formats else (
                    output_path.with_name(f"_tmp_subs_{output_path.stem}.ass")
                )
                ass_path.parent.mkdir(parents=True, exist_ok=True)
                ass_path.write_text(ass_content, encoding="utf-8")
                if "ass" in formats:
                    saved["ass"] = ass_path
                if want_burn:
                    burn_path = ass_path
        except Exception as exc:
            logger.warning("Не удалось собрать субтитры для {}: {}", output_path.name, exc)
            return burn_path if burn_path and burn_path.exists() else None, saved
        return burn_path, saved

    def _banner_obstacles(
        self,
        settings: UserSettings,
        crop_samples: list[CropSample],
        head_regions: list[BoundingBox | None],
        branding: BrandingOverlay | None,
        appear_sec: float,
        duration_sec: float,
    ) -> list[BoundingBox]:
        """Зоны выходного кадра (px), которые плашка не должна закрывать за время
        показа: голова животного (пересчитана из исходных координат через окно
        кропа каждого сэмпла) и логотип/Subscribe."""
        out_w, out_h = settings.export.width, settings.export.height
        obstacles: list[BoundingBox] = []

        for sample, head in zip(crop_samples, head_regions):
            if head is None or not (appear_sec <= sample.timestamp_sec <= appear_sec + duration_sec):
                continue
            window = sample.window
            crop_w, crop_h = window.x2 - window.x1, window.y2 - window.y1
            if crop_w <= 0 or crop_h <= 0:
                continue
            sx, sy = out_w / crop_w, out_h / crop_h
            # обе границы зажимаем в кадр: зона вне окна кропа схлопывается в пустой прямоугольник
            x1 = min(float(out_w), max(0.0, (head.x1 - window.x1) * sx))
            x2 = min(float(out_w), max(0.0, (head.x2 - window.x1) * sx))
            y1 = min(float(out_h), max(0.0, (head.y1 - window.y1) * sy))
            y2 = min(float(out_h), max(0.0, (head.y2 - window.y1) * sy))
            if x2 - x1 > 1 and y2 - y1 > 1:
                obstacles.append(BoundingBox(x1, y1, x2, y2))

        if branding is not None:
            obstacles += [
                BoundingBox(*map(float, rect))
                for rect in branding.obstacle_rects(appear_sec, appear_sec + duration_sec)
            ]
        return obstacles

    def _build_banner_rect(
        self,
        settings: UserSettings,
        crop_samples: list[CropSample],
        head_regions: list[BoundingBox | None],
        branding: BrandingOverlay | None,
        appear_sec: float,
        duration_sec: float,
        zone: SafeZone | None = None,
        subtitles_during_banner: bool = False,
    ) -> tuple[int, int, int, int]:
        """Прямоугольник плашки внутри безопасной зоны Shorts. Если в время показа плашки
        идут субтитры, нижняя граница зоны поднимается над полосой субтитров, поэтому они
        не пересекаются; дальше плашка обходит голову животного, логотип и Subscribe.
        """
        width, height = settings.export.width, settings.export.height
        zone = zone or SafeZone.from_settings((width, height), settings.safe_zone)

        bounds_zone = zone
        if subtitles_during_banner:
            band = estimate_subtitle_band_height(settings.subtitles.style_preset)
            bounds_zone = zone.with_bottom(zone.y2 - band - SUBTITLE_BANNER_GAP_PX)

        banner_width, banner_height = min(int(width * 0.85), bounds_zone.width), 220

        try:
            preferred_position = BannerPosition(settings.branding.banner_position)
        except ValueError:
            preferred_position = BannerPosition.BOTTOM_CENTER

        obstacles: list[BoundingBox] = []
        if settings.branding.collision_avoidance:
            obstacles = self._banner_obstacles(
                settings, crop_samples, head_regions, branding, appear_sec, duration_sec
            )

        resolution = resolve_banner_position_over_time(
            preferred_position, obstacles, width, height, banner_width, banner_height, bounds=bounds_zone.as_box()
        )
        if resolution.was_repositioned:
            logger.info(
                "Плашка сдвинута {} -> {}: обход головы животного/логотипа/Subscribe",
                preferred_position.value, resolution.final_position.value,
            )
        rect = resolution.final_rect
        return (int(rect.x1), int(rect.y1), int(rect.x2), int(rect.y2))
