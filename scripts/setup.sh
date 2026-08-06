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
set -euo pipefail

CONTAINER_NAME="memo-cat-studio"
CONTAINER_IMAGE="fedora:40"
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

    if distrobox list 2>/dev/null | grep -qw "$CONTAINER_NAME"; then
        ok "Контейнер '$CONTAINER_NAME' уже существует"
    else
        log "Создаю контейнер '$CONTAINER_NAME' (образ $CONTAINER_IMAGE, GPU проброшен)"
        distrobox create \
            --name "$CONTAINER_NAME" \
            --image "$CONTAINER_IMAGE" \
            --additional-flags "--device /dev/dri" \
            --yes
        ok "Контейнер создан"
    fi

    log "Захожу в контейнер и продолжаю установку там..."
    exec distrobox enter "$CONTAINER_NAME" -- bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
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
   ffmpeg ffmpeg-libs \
    python3 python3-devel python3-pip \
    libva libva-utils \
    vulkan-tools mesa-vulkan-drivers mesa-va-drivers \
    qt6-qtbase-devel qt6-qtmultimedia qt6-qtmultimedia-devel \
    gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free gstreamer1-libav
ok "Системные зависимости установлены"

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
if [ "${1:-}" != "--skip-models" ]; then
    log "Загружаю ML-модели (YOLO11 ONNX, faster-whisper) — можно пропустить флагом --skip-models"
    python scripts/download_models.py || warn "Загрузка моделей не удалась — повторить позже: python scripts/download_models.py"
else
    warn "Загрузка моделей пропущена (--skip-models)"
fi

# ------------------------------------------------------------
# ШАГ 8: запуск приложения
# ------------------------------------------------------------
log "Установка завершена. Запускаю Memo Cat AI Studio Pro..."
export QT_QPA_PLATFORM=xcb
export GST_PLUGIN_PATH=/usr/lib64/gstreamer-1.0:/usr/lib/gstreamer-1.0
# Принудительно указываем бэкенд
export QT_MULTIMEDIA_PREFERRED_BACKEND=ffmpeg
python -m ui.main_window
