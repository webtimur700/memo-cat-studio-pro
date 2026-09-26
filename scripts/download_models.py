#!/usr/bin/env python3
"""Подготовка ML-моделей проекта. Требует сети (в отличие от остального
приложения, которое после установки работает полностью офлайн).

Запуск: python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"


def export_yolo11_to_onnx() -> None:
    onnx_path = MODELS_DIR / "yolo11n.onnx"
    if onnx_path.exists():
        print(f"[skip] {onnx_path} уже существует")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[ошибка] Пакет 'ultralytics' не установлен. Он нужен ТОЛЬКО для "
            "экспорта модели в ONNX (не для инференса — инференс идёт через "
            "vision/yolo_detector.py на чистом onnxruntime). Установите: "
            "pip install ultralytics --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Скачивание весов YOLO11n и экспорт в ONNX...")
    model = YOLO("yolo11n.pt")
    exported_path = model.export(format="onnx")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Path(exported_path).rename(onnx_path)
    print(f"[ok] Сохранено: {onnx_path}")


def check_faster_whisper_cache() -> None:
    # faster-whisper скачивает и кэширует модель автоматически при первом
    # использовании (в ~/.cache/huggingface) — здесь только прогреваем кэш
    # заранее, чтобы первый реальный запуск в приложении не ждал загрузки.
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "[ошибка] Пакет 'faster-whisper' не установлен: "
            "pip install faster-whisper --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)

    # ALL_PROXY=socks://… (VPN-клиенты) роняет httpx в huggingface_hub ещё до запроса
    from core.proxy_env import make_httpx_safe

    for change in make_httpx_safe():
        print(f"[прокси] {change}")
    print("Прогрев кэша faster-whisper (модель 'small', int8, CPU)...")
    WhisperModel("small", device="cpu", compute_type="int8")
    print("[ok] Модель faster-whisper закэширована")


YAMNET_BASE_URL = "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main"   # Apache-2.0, конвертация Google YAMNet


def download_yamnet() -> None:
    """YAMNet (AudioSet, 521 класс: лай, мяуканье, смех...) в ONNX для audio/event_classifier.py."""
    import urllib.request

    target = MODELS_DIR / "yamnet"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("yamnet.onnx", "yamnet_class_map.csv"):
        path = target / name
        if path.exists():
            print(f"[skip] {path} уже существует")
            continue
        print(f"Скачивание {name}...")
        urllib.request.urlretrieve(f"{YAMNET_BASE_URL}/{name}", path)
        print(f"[ok] Сохранено: {path}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_yolo11_to_onnx()
    download_yamnet()
    check_faster_whisper_cache()
    print("\nВсе модели подготовлены. Можно запускать приложение офлайн.")


if __name__ == "__main__":
    main()
