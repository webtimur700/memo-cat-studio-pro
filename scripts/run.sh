#!/usr/bin/env bash
# Быстрый запуск УЖЕ УСТАНОВЛЕННОГО приложения (после scripts/setup.sh).
#
#   bash scripts/run.sh
#
# Отдельный скрипт нужен, потому что переменные окружения QT_QPA_PLATFORM /
# GST_PLUGIN_PATH / QT_MULTIMEDIA_PREFERRED_BACKEND (нужны для стабильной
# работы QtMultimedia/предпросмотра видео) задаются в scripts/setup.sh только
# на время его собственного запуска — обычный "python -m ui.main_window" в
# новом терминале их не увидит. Этот скрипт — та же финальная часть
# setup.sh, но без переустановки зависимостей при каждом запуске.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

# имя контейнера — как в setup.sh (MEMO_CAT_CONTAINER, по умолчанию memo-cat-studio)
CONTAINER_NAME="${MEMO_CAT_CONTAINER:-memo-cat-studio}"

if [ ! -f /run/.containerenv ]; then
    echo "Запускаю через Distrobox-контейнер $CONTAINER_NAME..."
    exec distrobox enter "$CONTAINER_NAME" -- bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

cd "$PROJECT_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Виртуальное окружение не найдено. Сначала выполните установку: bash scripts/setup.sh"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export QT_QPA_PLATFORM=xcb
export GST_PLUGIN_PATH=/usr/lib64/gstreamer-1.0:/usr/lib/gstreamer-1.0
export QT_MULTIMEDIA_PREFERRED_BACKEND=ffmpeg

python -m ui.main_window
