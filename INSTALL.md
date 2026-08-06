# Установка и запуск — Memo Cat AI Studio Pro

Инструкция рассчитана на целевую машину: **AMD Ryzen 7 8845HS / Radeon 780M /
32 GB RAM / Bazzite Linux (Fedora Atomic), GNOME**. На другом железе/дистрибутиве
основные шаги те же, отличия отмечены отдельно.

На момент этой инструкции реализованы и протестированы Шаги 1–6 (архитектура,
конфигурация, UI, Video Engine, AI Engine). Дальше в проекте появятся
`subtitles/`, `llm/`, `effects/`, `animation/`, `export/`, `queue/` — команды
запуска ниже уже сейчас позволяют проверить готовую часть (UI, приём видео,
детекцию сцен, YOLO-детекцию, трекинг, автокадрирование).

---

## 1. Предварительные требования на хосте

Bazzite — иммутабельный дистрибутив (Fedora Atomic), поэтому пакеты **не**
ставятся через `rpm-ostree install` в базовую систему — используем Distrobox
(решение, зафиксированное на Шаге 3).

```bash
# Distrobox обычно уже предустановлен в Bazzite. Проверить:
distrobox --version

# Если нет — установить как layered-пакет (единственное системное изменение,
# требуется один раз и одна перезагрузка):
rpm-ostree install distrobox
systemctl reboot
```

Проверить, что Podman (движок Distrobox) на месте — он у тебя уже используется
для [tgbot], так что должен быть настроен:

```bash
podman --version
```

## 2. LM Studio (на хосте, НЕ в контейнере)

LM Studio уже установлена. Убедиться, что локальный API-сервер включён:

1. Открыть LM Studio → вкладка **Developer** (или **Local Server**).
2. Загрузить любую из уже скачанных моделей (Qwen3.6 35B A3B / GPT-OSS 20B /
   Qwen2.5 14B / Gemma 4 E4B / Qwen2.5 7B).
3. Нажать **Start Server** — по умолчанию поднимется `http://localhost:1234/v1`.

Проверить с хоста:

```bash
curl http://localhost:1234/v1/models
```

Должен вернуться JSON со списком загруженных моделей. Это тот же адрес, что
уже прописан в `.env.example` (`LM_STUDIO_BASE_URL`).

## 3. Создание Distrobox-контейнера

```bash
cd memo-cat-studio-pro

# Создать контейнер с проброшенным GPU и Wayland/X11
distrobox create \
  --name memo-cat-studio \
  --image fedora:40 \
  --additional-flags "--device /dev/dri" \
  --home ~/.distrobox/memo-cat-studio

distrobox enter memo-cat-studio
```

Внутри контейнера (после `distrobox enter`):

```bash
# Системные зависимости (ffmpeg с VAAPI-поддержкой, библиотеки сборки Qt)
sudo dnf install -y \
    ffmpeg ffmpeg-libs \
    python3.12 python3.12-devel python3-pip \
    libva libva-utils vulkan-tools \
    mesa-vulkan-drivers mesa-va-drivers \
    qt6-qtbase-devel

# Честная проверка, что видит система на этой машине —
# результат этой команды определит, какой compute_profile реально имеет смысл
ffmpeg -hide_banner -hwaccels
vainfo
vulkaninfo --summary
```

## 4. Python-окружение и зависимости проекта

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

## 5. Загрузка ML-моделей (одна команда, но реально стягивает веса из сети)

```bash
python scripts/download_models.py
```

Ожидаемый результат — в `models/` появляются:
- `yolo11n.onnx` (детекция кошек/собак/людей)
- веса `faster-whisper` (модель `small`, см. `config/default_settings.yaml`)
- `silero_vad.onnx`

Если скрипта ещё нет в твоей копии проекта (появится на Шаге 11 "Export
Engine"/утилиты) — временная альтернатива вручную:

```bash
pip install ultralytics --no-deps  # только для экспорта, не для инференса
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx')"
mkdir -p models && mv yolo11n.onnx models/
```

## 6. Диагностика вычислительного backend'а

Перед первым запуском стоит честно проверить, что реально доступно на этой
машине — результат определяет `COMPUTE_PROFILE` в `.env`:

```bash
python -c "
from vision.compute_backend import ComputeBackend
backend = ComputeBackend()
resolved = backend.resolve()
print('Активный provider:', resolved.active_provider)
print('Приоритет:', resolved.provider_priority)
print('Диагностика:', resolved.diagnostics)
"
```

На AMD 780M реалистично ожидать `active_provider: CPUExecutionProvider` (см.
честное объяснение в `config/compute_profiles.yaml`) — если увидишь Vulkan/ROCm
в списке `provider_priority`, это будет означать, что твоя сборка
`onnxruntime` действительно скомпилирована с этими EP, что не гарантировано
стандартным pip-пакетом.

## 7. Настройка окружения проекта

```bash
cp .env.example .env
# при необходимости поправить LM_STUDIO_BASE_URL / COMPUTE_PROFILE
```

## 8. Проверка Video Engine на своём файле

```bash
python -c "
from pathlib import Path
from video.ingestion_service import IngestionService

service = IngestionService()
source = service.ingest(Path('/путь/к/твоему/видео.mp4'))
print(source)
"
```

## 9. Запуск приложения

```bash
python -m ui.main_window
# или, после pip install -e .:
memo-cat-studio
```

Первый запуск GUI-приложения из Distrobox-контейнера в GNOME (Wayland) должен
подхватить дисплей автоматически — Distrobox пробрасывает `$WAYLAND_DISPLAY`/
`$DISPLAY` и `/tmp/.X11-unix` из хоста. Если окно не появляется — проверить:

```bash
echo $WAYLAND_DISPLAY $DISPLAY
```

## 10. Экспорт приложения в меню GNOME (опционально, для запуска не из терминала)

```bash
distrobox-export --app memo-cat-studio
```
После этого в меню приложений GNOME появится обычная иконка "Memo Cat AI
Studio Pro", запускающая контейнер прозрачно для пользователя.

---

## Troubleshooting

| Симптом | Причина / что проверить |
|---|---|
| `ModuleNotFoundError: PySide6` | Venv не активирован (`source .venv/bin/activate`), либо `pip install -e .` не выполнялся |
| GUI не открывается из контейнера | `--device /dev/dri` не был указан при `distrobox create`; либо `$WAYLAND_DISPLAY` не проброшен — пересоздать контейнер |
| `LM Studio` недоступна по `localhost:1234` | Сервер не запущен во вкладке Developer/Local Server LM Studio на хосте; либо порт занят другим процессом |
| `ffprobe`/`ffmpeg` не найдены | Не установлен пакет `ffmpeg` внутри контейнера (шаг 3) |
| `scenedetect` не установлен | `pip install scenedetect[opencv]` — не входит в базовый `pyproject.toml` минимальный набор на момент Шага 5, будет закреплён окончательно на Шаге 12 (Documentation) |
| `ComputeBackend` показывает только CPU | Ожидаемо на большинстве стандартных сборок `onnxruntime` — см. `config/compute_profiles.yaml`. Не баг, честное поведение |

---

## Быстрая самопроверка после установки (smoke test)

```bash
python -c "
import onnxruntime, cv2, mediapipe
from PySide6.QtWidgets import QApplication
print('onnxruntime:', onnxruntime.__version__)
print('opencv:', cv2.__version__)
print('mediapipe:', mediapipe.__version__)
print('PySide6 импортируется без ошибок — Qt-зависимости на месте')
"
```

Если все три строки напечатались без traceback — окружение готово к работе с
частью проекта, реализованной на данный момент (Шаги 1–6).
