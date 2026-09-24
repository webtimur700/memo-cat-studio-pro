# Changelog

Все версии до первого стабильного релиза документируются по шагам разработки
(не по датам — проект строится непрерывно в рамках одной сессии
проектирования).

## [Unreleased]

### Добавлено — очередь пакетной обработки
- `pipeline/job_scheduler.py`: по умолчанию одно видео за раз (`batch.max_concurrent_videos`, было 100), остальные
  ждут со статусом «В очереди»; настройка «Видео одновременно» в экране настроек
- `pipeline/shared_models.py`: YOLO и Whisper грузятся один раз на очередь; LLM выгружается, только когда очередь пуста
- Очередь на 1000+ видео не тормозит интерфейс (делегат прогресса вместо виджетов, фиксированные колонки)
- Исправлено: зависание интерфейса при смене клипа в плеере после появления результатов второго видео
- Одно и то же видео дважды в очереди не добавляется; не запустившаяся задача помечается ошибкой, очередь идёт дальше
- Проверка на реальном видео: `docs/queue_verification.md`, `scripts/verify_queue.py`

### Добавлено — Шаг 14: Документация
- README.md, API.md, CHANGELOG.md (этот файл)

### Добавлено — Шаг 13: Тесты
- 43 unit/integration теста (pytest), покрывающие: settings, tracker,
  pose motion analyzer, smart crop, ASS-рендер, collision detector,
  анимации, LLM-парсинг заголовков/хештегов, video engine, export pipeline,
  LM Studio provider, plugin loader
- Все тесты подтверждены реальным прогоном в процессе разработки

### Добавлено — Шаг 12: Installer
- `scripts/install.sh` — установка одной командой (venv, зависимости,
  диагностика GPU, .env, модели)
- `scripts/download_models.py` — экспорт YOLO11 в ONNX, прогрев кэша
  faster-whisper
- `scripts/check_gpu_support.py` — самостоятельная диагностика
  вычислительного backend'а

### Добавлено — Шаг 11: Plugin SDK
- `core/interfaces/plugin_base.py`, `plugins/sdk/registry.py`,
  `plugins/loader.py`
- Рабочий пример плагина (`vintage_sepia` эффект)
- Устойчивость к сломанным плагинам (один сбой не роняет загрузку остальных)

### Добавлено — Шаг 10: Export Engine + LLM
- `export/dynamic_crop.py` — покадровое автокадрирование через ffmpeg
  crop-фильтр с кусочно-линейной интерполяцией
- `export/export_service.py` — полная сборка Shorts (crop + субтитры + overlay
  плашки)
- `llm/lm_studio_provider.py` — клиент к LM Studio (stdlib `urllib`, без
  лишних зависимостей)
- `llm/prompts/` — генерация заголовков/описания/хештегов

**Исправлено:**
- `export_service.py` не ограничивал длительность выходного клипа длительностью
  плана (утекал на всю длину исходника) — добавлен `-t {clip_duration}`
- WebM/VP9 (`libvpx-vp9`) в используемой сборке ffmpeg заявляет поддержку
  `yuva420p`, но реально теряет альфа-канал при кодировании — заменено на
  `qtrle`/MOV, подтверждено round-trip тестом (кодирование → декодирование →
  проверка альфа-пикселей)
- `titles_prompt.parse_titles()` не распознавал нумерацию с пробелом перед
  разделителем ("3 - текст") — расширена регулярка

### Добавлено — Шаг 9: Animation Engine
- `animation/easing.py` — стандартные easing-функции
- `animation/promo_banner_animator.py` — enter/hold/exit анимация плашки
  (slide+fade+scale+glow → floating → fade+slide)
- `animation/logo_animator.py`, `animation/subscribe_button_animator.py`

### Добавлено — Шаг 8: Effects Engine
- `effects/glassmorphism_renderer.py` — рендер плашки (blur+glow+rounded+shadow)
- `effects/cover_generator.py` — обложка с текстом и эмодзи
- `effects/font_utils.py` — проверка покрытия символов шрифтом через fontTools
- `effects/collision_detector.py` — репозиционирование плашки при пересечении
  с животным

**Исправлено:**
- Шрифт Poppins-Bold не содержит кириллицу — добавлена честная проверка
  покрытия символов (`font_supports_text`) вместо визуальной проверки на глаз
- NotoColorEmoji — bitmap-шрифт с фиксированным набором размеров, не принимает
  произвольный `size` — добавлен рендер через тайлы с последующим `resize()`
- Геометрическая эвристика "головы животного" блокировала смещение в одну из
  сторон из-за неверного порядка `min()/max()` при клэмпинге

### Добавлено — Шаг 7: Subtitle Engine
- `subtitles/subtitle_service.py` — faster-whisper, группировка слов в сегменты
- `subtitles/ass_renderer.py` — ASS/SRT рендер, karaoke-подсветка слова,
  4 стилевых пресета

### Добавлено — Шаг 6: AI Engine
- `vision/compute_backend.py` — честный резолв Vulkan/ROCm/OpenCL/CPU
- `vision/yolo_detector.py`, `vision/tracker.py`, `vision/pose_motion_analyzer.py`,
  `vision/mediapipe_face.py`, `vision/smart_crop.py`

### Добавлено — Шаг 5: Video Engine
- `video/ingestion_service.py`, `video/ffmpeg_wrapper.py`,
  `video/scene_detector.py`, `video/frame_extractor.py`
- `INSTALL.md`

### Добавлено — Шаг 4: UI
- PySide6: `main_window.py`, `project_view.py`, `timeline_view.py`,
  `preview_player.py`, `batch_queue_view.py`, `settings_view.py`,
  glassmorphism-виджеты, тёмная тема

### Добавлено — Шаг 3: Конфигурация
- `pyproject.toml`, `config/default_settings.yaml`,
  `config/compute_profiles.yaml`, `config/logging.yaml`, `.env.example`

### Добавлено — Шаги 1-2: Архитектура и структура
- Clean Architecture, выбор технологического стека под целевое железо
  (AMD Ryzen 7 8845HS / Radeon 780M / Bazzite Linux)
- Полная структура каталогов проекта
