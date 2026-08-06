#!/usr/bin/env python3
"""Честная диагностика вычислительных возможностей текущей машины.
Использует ту же логику, что и vision/compute_backend.py и
video/ffmpeg_wrapper.is_vaapi_available() — этот скрипт нужен, чтобы увидеть
результат ДО запуска самого приложения, при установке.

Запуск: python scripts/check_gpu_support.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video.ffmpeg_wrapper import is_vaapi_available  # noqa: E402
from vision.compute_backend import ComputeBackend  # noqa: E402


def main() -> None:
    print("=== Memo Cat AI Studio Pro — диагностика вычислительного backend'а ===\n")

    backend = ComputeBackend()
    resolved = backend.resolve()

    print(f"ML-инференс (ONNX Runtime):")
    print(f"  Активный провайдер: {resolved.active_provider}")
    print(f"  Приоритет провайдеров: {resolved.provider_priority}")
    print(f"  Потоков CPU выделено: {resolved.cpu_threads}")
    print(f"  Диагностика:")
    for key, value in resolved.diagnostics.items():
        status = "OK" if value else "недоступно"
        print(f"    - {key}: {status}")

    print(f"\nВидео-декодирование/энкодирование (FFmpeg):")
    vaapi_ok = is_vaapi_available()
    print(f"  VAAPI: {'доступен' if vaapi_ok else 'недоступен (будет использован software-путь)'}")

    print("\n=== Итог ===")
    if resolved.active_provider == "CPUExecutionProvider":
        print(
            "ML-инференс будет идти на CPU. Это ОЖИДАЕМО для большинства сборок "
            "onnxruntime на AMD iGPU (см. честное объяснение в config/compute_profiles.yaml) "
            "и не является ошибкой установки."
        )
    else:
        print(f"ML-инференс будет использовать GPU через {resolved.active_provider}.")


if __name__ == "__main__":
    main()
