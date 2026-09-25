# Memo Cat AI Studio Pro

Десктоп-приложение (PySide6, Python 3.12) для автоматической нарезки
YouTube Shorts из длинных видео с животными: поиск интересных моментов,
кадрирование в 9:16 со слежением за животным, субтитры, анимированная
рекламная плашка 100zarplat.ru, логотип и кнопка Subscribe, фоновая музыка,
заголовки, описание и хештеги через локальную LLM, экспорт MP4
(H264/AAC, 30 fps, 1080x1920).

## Окружение

- Хост: Bazzite (Fedora Atomic, иммутабельная система), AMD Ryzen 7 8845HS,
  Radeon 780M, 32 ГБ ОЗУ, общая с iGPU.
- Приложение работает только внутри Distrobox-контейнера `memo-cat-studio`
  (образ fedora:40): `distrobox enter memo-cat-studio`. На хосте нет
  python3.12, пакеты на хост не ставим.
- Виртуальное окружение: `.venv` в корне проекта.
- LM Studio (Flatpak) запущена на хосте, из контейнера доступна по
  http://localhost:1234. На машине могут работать ollama и open-webui и
  занимать память.
- В окружении бывает системный прокси (`ALL_PROXY=socks://...`,
  `no_proxy` с CIDR `127.0.0.0/8`). Запросы к LM Studio идут в обход прокси,
  faster-whisper сначала грузится из локального кэша. Не ломай это.
- GPU-ускорение ML недоступно (onnxruntime собран без Vulkan и ROCm),
  работает CPUExecutionProvider. Это ограничение сборки, а не баг.

## Команды

- Установка: `bash scripts/setup.sh` (запускается с хоста, сам заходит в контейнер)
- Запуск: `bash scripts/run.sh`
- Модели: `python scripts/download_models.py` (YOLO11 ONNX, faster-whisper, YAMNet)
- Тесты: `pytest tests/ -v`, после любых изменений должны быть зелёными
- Результаты: `export/output/`: клипы `*_momentN.mp4`, обложки `*_cover.png`,
  `*.json` (10 заголовков, описание, хештеги, транскрипция), `*.srt`, `*.ass`
- Логи: `logs/app.log` (там полные ffmpeg-команды и их stderr)

## Устройство

- `pipeline/pipeline_runner.py`: оркестратор (ingestion → скоринг → отбор
  моментов → кроп → субтитры → плашка, логотип, Subscribe → музыка → экспорт).
  Пакет называется `pipeline`, а не `queue`: `queue` — модуль стандартной
  библиотеки, конфликт имён ломает импорты.
- `ui/pipeline_worker.py`: запуск пайплайна в QThread. Очередь в
  `ui/main_window.py`, по умолчанию одно видео за раз. YOLO, Whisper и LLM
  грузятся один раз на очередь, LLM выгружается, когда очередь пуста.
- `llm/model_selector.py`, `llm/managed_provider.py`: выбор модели по
  рейтингу из `docs/llm_benchmark.md` и свободной памяти.
  `LM_STUDIO_MODEL_OVERRIDE` в `.env` имеет приоритет.
- Остальное: `scoring/`, `cutting/`, `vision/`, `subtitles/`, `effects/`,
  `animation/`, `export/`, `plugins/`, `core/`.
- ML-модели и LM Studio могут быть недоступны. Пайплайн должен честно
  деградировать с warning в логе, а не падать.

## Известные грабли (уже исправлены, не возвращай)

- Пути к шрифтам: система Fedora, а не Debian. Шрифты ищутся через
  fontconfig. Системный Noto Color Emoji (COLRv1) Pillow рисует пустым,
  поэтому эмодзи рисуются через Twemoji.
- ffmpeg: `-ss` стоит перед `-i` (`ExportPlan.source_start_sec`). Путь к
  `.ass` передаётся через cwd процесса, без экранирования в filter_chain.
- Прозрачный overlay кодируется в qtrle/MOV: libvpx-vp9 в этой сборке
  теряет альфа-канал.
- Язык речи определяется автоматически (не форсировать "ru"), субтитры
  переводятся на русский через LLM.
- Рассуждающие модели в LM Studio: блок `<think>` вырезается, используется
  reasoning_effort=none.
- QMediaPlayer.setSource мог взаимно блокироваться с потоком декодера
  через GIL. На это есть регрессионный тест.
- Время в SRT/ASS: миллисекунды не должны округляться до 1000
  (YouTube отвергает такой SRT).
- Музыка из `assets/music/` в git не попадает (права на треки).

## Где искать подробности

- `CHANGELOG.md`, `git log --oneline -40`
- `docs/`: llm_benchmark, moment_selection, whisper_benchmark, audio_events,
  queue_verification

## Открытые задачи

На данный момент все пункты первоначального списка выполнены (см. `git log`).
Актуальная работа: скорость обработки, см. `docs/performance.md`.

## Правила работы

- Проверяй изменения на реальном видео (`~/Видео/myvideo.mp4`, для быстрых
  итераций вырезай кусок через ffmpeg), смотри кадры глазами, проверяй
  клипы через ffprobe.
- Ищи причину ошибки по `logs/app.log`, а не по догадкам.
- После каждого законченного пункта делай git commit.
- В конце работы пиши отчёт: что сделано, как проверено, что не проверено
  и почему.
- В конце сессии, когда тесты зелёные, отправляй коммиты: git push origin main.
