#!/usr/bin/env bash
# Единый скрипт установки и запуска Memo Cat AI Studio Pro.
#
# Запускать ОДНОЙ командой с хоста Bazzite (не важно, внутри контейнера
# вы уже или нет — скрипт сам разберётся):
#
#   bash scripts/setup.sh
#
# Что делает:
#   1. Если запущен на голом хосте — создаёт (если нет) Distrobox-контейнер
#      "memo-cat-studio" и перезапускает сам себя уже внутри него.
#   2. Внутри контейнера — подключает RPM Fusion, ставит системные пакеты
#      (ffmpeg, python3.12, Vulkan/VAAPI, Qt6), создаёт venv, ставит Python-
#      зависимости, готовит .env.
#   3. Запускает приложение.
#
# Скрипт идемпотентен — его можно запускать повторно, уже готовые шаги будут
# пропущены (используйте это же, если что-то прервалось на середине).
#
# Параметры:
#   --skip-models   не скачивать ML-модели (YOLO, Whisper, YAMNet)
#   --no-run        только установка, приложение не запускать
# Переменные окружения:
#   MEMO_CAT_CONTAINER   имя контейнера (по умолчанию memo-cat-studio); run.sh читает ту же переменную
#   MEMO_CAT_IMAGE       образ контейнера (по умолчанию fedora:40)
#   MEMO_CAT_CONTAINER_HOME  отдельный домашний каталог контейнера (distrobox --home); по умолчанию — общий с хостом
set -euo pipefail

CONTAINER_NAME="${MEMO_CAT_CONTAINER:-memo-cat-studio}"
CONTAINER_IMAGE="${MEMO_CAT_IMAGE:-fedora:40}"
CONTAINER_HOME="${MEMO_CAT_CONTAINER_HOME:-}"
SKIP_MODELS=0
NO_RUN=0
for arg in "$@"; do
    case "$arg" in
        --skip-models) SKIP_MODELS=1 ;;
        --no-run) NO_RUN=1 ;;
        *) echo "Неизвестный параметр: $arg (есть --skip-models, --no-run)"; exit 2 ;;
    esac
done
PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

log()  { echo -e "\n\033[1;35m==> $1\033[0m"; }
ok()   { echo -e "\033[0;32m[ok]\033[0m $1"; }
warn() { echo -e "\033[0;33m[!]\033[0m $1"; }

# ============================================================
# ШАГ 0: если мы на голом хосте — создать контейнер и зайти в него
# ============================================================
if [ ! -f /run/.containerenv ]; then
    log "Обнаружен запуск с хоста — потребуется Distrobox-контейнер"

    if ! command -v distrobox >/dev/null 2>&1; then
        echo "Distrobox не найден. На Bazzite он обычно предустановлен."
        echo "Если нет — установите и перезагрузитесь:"
        echo "    rpm-ostree install distrobox && systemctl reboot"
        exit 1
    fi
    ok "Distrobox найден"

    # имя сравниваем целиком: `grep -w` принял бы "memo-cat-studio-test" за "memo-cat-studio" ('-' не буква)
    if distrobox list --no-color 2>/dev/null | awk -F'|' 'NR>1 {gsub(/^ +| +$/, "", $2); print $2}' | grep -qxF "$CONTAINER_NAME"; then
        ok "Контейнер '$CONTAINER_NAME' уже существует"
    else
        log "Создаю контейнер '$CONTAINER_NAME' (образ $CONTAINER_IMAGE, GPU проброшен)"
        create_args=(--name "$CONTAINER_NAME" --image "$CONTAINER_IMAGE" --additional-flags "--device /dev/dri" --yes)
        if [ -n "$CONTAINER_HOME" ]; then
            create_args+=(--home "$CONTAINER_HOME")
        fi
        distrobox create "${create_args[@]}"
        ok "Контейнер создан"
    fi

    log "Захожу в контейнер и продолжаю установку там..."
    exec distrobox enter "$CONTAINER_NAME" -- env MEMO_CAT_CONTAINER="$CONTAINER_NAME" bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# ============================================================
# С этой точки — мы ТОЧНО внутри контейнера
# ============================================================
cd "$PROJECT_ROOT"
log "Работаю внутри контейнера, директория проекта: $PROJECT_ROOT"

# ------------------------------------------------------------
# ШАГ 1: RPM Fusion (нужен для ffmpeg — он не в стандартных репах Fedora)
# ------------------------------------------------------------
if dnf repolist 2>/dev/null | grep -qi rpmfusion; then
    ok "RPM Fusion уже подключён"
else
    log "Подключаю RPM Fusion (нужен для ffmpeg)"
    sudo dnf install -y \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm"
    ok "RPM Fusion подключён"
fi

# ------------------------------------------------------------
# ШАГ 2: системные пакеты
# ------------------------------------------------------------
log "Устанавливаю системные зависимости (ffmpeg, Python 3.12, Vulkan/VAAPI, Qt6)"
sudo dnf install -y \
    ffmpeg ffmpeg-libs libatomic \
    python3.12 python3.12-devel python3-pip \
    libva libva-utils \
    vulkan-tools mesa-vulkan-drivers \
    qt6-qtbase-devel qt6-qtmultimedia qt6-qtmultimedia-devel \
    gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free gstreamer1-libav
ok "Системные зависимости установлены"

# ------------------------------------------------------------
# ШАГ 2b: аппаратное кодирование H.264 (VAAPI, Radeon). Штатный mesa-va-drivers в Fedora собран
# без H.264/HEVC (патенты), поэтому нужен mesa-va-drivers-freeworld из RPM Fusion. Без него экспорт
# автоматически идёт через libx264 (в 4-5 раз медленнее), поэтому сбой здесь не фатален.
# ------------------------------------------------------------
if rpm -q mesa-va-drivers-freeworld >/dev/null 2>&1; then
    ok "mesa-va-drivers-freeworld уже установлен"
elif rpm -q mesa-va-drivers >/dev/null 2>&1; then
    log "Заменяю mesa-va-drivers на mesa-va-drivers-freeworld (H.264 через VAAPI)"
    sudo dnf swap -y mesa-va-drivers mesa-va-drivers-freeworld || warn "Не удалось заменить драйвер VAAPI — экспорт будет через libx264"
else
    sudo dnf install -y mesa-va-drivers-freeworld || warn "Не удалось установить mesa-va-drivers-freeworld — экспорт будет через libx264"
fi
if vainfo 2>/dev/null | grep -q "VAProfileH264.*VAEntrypointEncSlice"; then
    ok "VAAPI: аппаратное кодирование H.264 доступно (VAEntrypointEncSlice)"
else
    warn "VAAPI: кодирования H.264 нет (vainfo не показывает VAEntrypointEncSlice для H264) — экспорт пойдёт через libx264"
fi

# ------------------------------------------------------------
# ШАГ 3: виртуальное окружение (строго внутри папки проекта)
# ------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    ok "Виртуальное окружение уже создано"
else
    log "Создаю виртуальное окружение (.venv)"
    python3.12 -m venv .venv
    ok "Создано"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
ok "Активно: $(which python)"

# ------------------------------------------------------------
# ШАГ 4: Python-зависимости
# ------------------------------------------------------------
log "Устанавливаю Python-зависимости проекта (может занять несколько минут)"
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
ok "Python-зависимости установлены"

# ------------------------------------------------------------
# ШАГ 5: .env
# ------------------------------------------------------------
if [ -f ".env" ]; then
    ok ".env уже существует"
else
    cp .env.example .env
    ok "Создан .env из .env.example"
fi

# ------------------------------------------------------------
# ШАГ 6: диагностика GPU-backend'а (информативно, не блокирует установку)
# ------------------------------------------------------------
log "Диагностика вычислительного backend'а этой машины"
python scripts/check_gpu_support.py || warn "Диагностика завершилась с предупреждением — не критично"

# ------------------------------------------------------------
# ШАГ 7: модели (требует сети; при ошибке — не блокируем запуск UI)
# ------------------------------------------------------------
if [ "$SKIP_MODELS" -eq 0 ]; then
    log "Загружаю ML-модели (YOLO11 ONNX, faster-whisper) — можно пропустить флагом --skip-models"
    python scripts/download_models.py || warn "Загрузка моделей не удалась — повторить позже: python scripts/download_models.py"
else
    warn "Загрузка моделей пропущена (--skip-models)"
fi

# ------------------------------------------------------------
# ШАГ 8: запуск приложения
# ------------------------------------------------------------
if [ "$NO_RUN" -eq 1 ]; then
    log "Установка завершена (--no-run). Запуск: bash scripts/run.sh"
    exit 0
fi
log "Установка завершена. Запускаю Memo Cat AI Studio Pro..."
export QT_QPA_PLATFORM=xcb
export GST_PLUGIN_PATH=/usr/lib64/gstreamer-1.0:/usr/lib/gstreamer-1.0
# Принудительно указываем бэкенд
export QT_MULTIMEDIA_PREFERRED_BACKEND=ffmpeg
python -m ui.main_window
