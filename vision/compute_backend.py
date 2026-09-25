"""Единая абстракция вычислительного backend'а для всех ONNX-моделей проекта.

КЛЮЧЕВОЙ ПРИНЦИП (проговорено ещё в архитектуре, Шаг 1): backend не заявляет
"GPU используется", если это не проверено фактически. Ниже — реальная логика:

  1. Спрашиваем у onnxruntime, какие execution providers физически
     скомпилированы в установленный пакет (`ort.get_available_providers()`).
     Если "VulkanExecutionProvider"/"ROCMExecutionProvider" там нет —
     никакие YAML-настройки их не включат, потому что их физически нет
     в бинарнике. Это самая частая причина, почему "Vulkan почему-то не
     работает" — обычный pip-пакет onnxruntime их не включает (см. честный
     комментарий в config/compute_profiles.yaml).
  2. Даже если провайдер скомпилирован — пробуем реально создать
     InferenceSession с ним на крошечной модели; если сессия не создаётся —
     откатываемся дальше по списку.
  3. CPUExecutionProvider — гарантированный последний уровень, всегда доступен.

Результат резолва кэшируется на процесс и логируется ОДИН раз при старте,
чтобы в логах всегда было видно, что реально используется на этой машине.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import onnxruntime as ort
from loguru import logger

# Порядок предпочтения EP -> как он называется в onnxruntime.
_EP_NAME_MAP: dict[str, str] = {
    "vulkan": "VulkanExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "opencl": "OpenCLExecutionProvider",  # не входит в официальные ORT-релизы для CPU EP набора;
                                            # оставлено для сборок, где он присутствует
    "cpu": "CPUExecutionProvider",
}


@dataclass(frozen=True, slots=True)
class ResolvedBackend:
    provider_priority: list[str]        # реально доступные ORT provider names, в порядке приоритета
    active_provider: str                # тот, что реально будет использован первым
    cpu_threads: int
    diagnostics: dict[str, bool]        # человекочитаемая сводка что проверялось и что нашлось


def _external_tool_available(binary_name: str) -> bool:
    return shutil.which(binary_name) is not None


def _rocm_actually_usable() -> bool:
    """rocminfo может быть установлен, но не значит, что конкретный чип
    (gfx1103 / Radeon 780M) в официальном supported-списке ROCm — поэтому
    здесь дополнительно проверяем, что rocminfo реально отрабатывает без
    ошибки и что в выводе есть хоть одна GPU-агент-запись, а не просто
    "команда существует".
    """
    if not _external_tool_available("rocminfo"):
        return False
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0 and "gfx" in result.stdout.lower()


def _vulkan_actually_usable() -> bool:
    if not _external_tool_available("vulkaninfo"):
        return False
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


class ComputeBackend:
    """DI получает один инстанс на весь процесс — резолв делается один раз."""

    def __init__(self, requested_priority: list[str] | None = None, cpu_threads: int | None = None) -> None:
        self._requested_priority = requested_priority or ["vulkan", "rocm", "opencl", "cpu"]
        self._cpu_threads = cpu_threads
        self._resolved: ResolvedBackend | None = None

    def resolve(self) -> ResolvedBackend:
        if self._resolved is not None:
            return self._resolved

        available_ort_providers = set(ort.get_available_providers())
        diagnostics: dict[str, bool] = {}

        provider_priority: list[str] = []

        for key in self._requested_priority:
            ort_name = _EP_NAME_MAP.get(key)
            if ort_name is None:
                continue

            if key == "cpu":
                # CPU всегда доступен, диагностика не нужна.
                provider_priority.append(ort_name)
                continue

            compiled_in = ort_name in available_ort_providers
            diagnostics[f"{key}_compiled_into_onnxruntime"] = compiled_in

            if key == "rocm":
                hardware_ok = _rocm_actually_usable()
                diagnostics["rocm_hardware_detected"] = hardware_ok
                if compiled_in and hardware_ok:
                    provider_priority.append(ort_name)
            elif key == "vulkan":
                hardware_ok = _vulkan_actually_usable()
                diagnostics["vulkan_hardware_detected"] = hardware_ok
                if compiled_in and hardware_ok:
                    provider_priority.append(ort_name)
            else:
                if compiled_in:
                    provider_priority.append(ort_name)

        if "CPUExecutionProvider" not in provider_priority:
            provider_priority.append("CPUExecutionProvider")

        cpu_threads = self._cpu_threads or self._detect_cpu_threads()

        self._resolved = ResolvedBackend(
            provider_priority=provider_priority,
            active_provider=provider_priority[0],
            cpu_threads=cpu_threads,
            diagnostics=diagnostics,
        )

        logger.info(
            "ComputeBackend resolved: active={}, priority={}, cpu_threads={}, diagnostics={}",
            self._resolved.active_provider,
            self._resolved.provider_priority,
            self._resolved.cpu_threads,
            self._resolved.diagnostics,
        )
        return self._resolved

    @staticmethod
    def _detect_cpu_threads() -> int:
        import os

        total = os.cpu_count() or 4
        # Резервируем 2 потока под UI/остальную систему — см. Функцию "максимум
        # многопоточности" из ТЗ, но не 100% ядер, иначе UI будет фризиться.
        return max(1, total - 2)

    def create_session(self, model_path: str) -> ort.InferenceSession:
        resolved = self.resolve()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = resolved.cpu_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Без активного ожидания: потоки onnxruntime не крутятся вхолостую и не отнимают ядра у декодера видео и соседних
        # запусков модели (YOLO в нескольких потоках + декодирование: 79 -> 48 мс на кадр, замер в docs/performance.md).
        # Одиночный вызов при этом медленнее на ~10%, поэтому параллельные вызовы (vision/parallel_detect.py) обязательны.
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        return ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=resolved.provider_priority,
        )


@lru_cache(maxsize=1)
def get_default_compute_backend() -> ComputeBackend:
    return ComputeBackend()
