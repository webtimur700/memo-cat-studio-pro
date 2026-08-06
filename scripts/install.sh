#!/usr/bin/env bash
# Установка Memo Cat AI Studio Pro одной командой.
# Предполагается запуск ВНУТРИ Distrobox-контейнера (см. INSTALL.md, шаги 1-3) —
# этот скрипт не трогает системные пакеты хоста, только Python-окружение проекта.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memo Cat AI Studio Pro — установка ==="

# --- 1. Проверка системных зависимостей ---
missing=()
for cmd in python3.12 ffmpeg ffprobe; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        missing+=("$cmd")
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo "Отсутствуют системные зависимости: ${missing[*]}"
    echo "См. INSTALL.md, раздел 3 (установка внутри Distrobox-контейнера)."
    exit 1
fi
echo "[ok] Системные зависимости на месте (python3.12, ffmpeg, ffprobe)"

# --- 2. Виртуальное окружение ---
if [ ! -d ".venv" ]; then
    echo "Создание виртуального окружения..."
    python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[ok] Виртуальное окружение активно: $(which python)"

# --- 3. Python-зависимости ---
echo "Установка Python-зависимостей (может занять несколько минут)..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
echo "[ok] Зависимости установлены"

# --- 4. Диагностика GPU/вычислительного backend'а ---
echo ""
echo "Диагностика вычислительных возможностей этой машины:"
python scripts/check_gpu_support.py

# --- 5. .env ---
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[ok] Создан .env из .env.example — проверьте LM_STUDIO_BASE_URL при необходимости"
fi

# --- 6. Модели (требует сети, можно пропустить флагом --skip-models) ---
if [ "${1:-}" != "--skip-models" ]; then
    echo ""
    echo "Загрузка ML-моделей (YOLO11 ONNX, faster-whisper)..."
    python scripts/download_models.py || echo "[предупреждение] Загрузка моделей не удалась — можно повторить позже: python scripts/download_models.py"
else
    echo "[skip] Загрузка моделей пропущена (--skip-models)"
fi

echo ""
echo "=== Установка завершена ==="
echo "Запуск приложения: source .venv/bin/activate && python -m ui.main_window"
