"""LLM-провайдер, который сам выбирает и загружает модель в LM Studio.

Порядок при первом обращении (можно заранее начать в фоне — start_loading_in_background —
пока пайплайн анализирует видео и кодирует первый клип):
  1. список моделей с метаданными (llm/lm_studio_api.py);
  2. выбор по рейтингу и свободной памяти (llm/model_selector.py), причина — в лог;
  3. чужие загруженные модели выгружаются (память нужна пайплайну), выбранная
     загружается через REST с небольшим контекстом (а не 262144 по умолчанию);
  4. запросы идут в LMStudioProvider с найденным reasoning_effort и флагом vision.
Если LM Studio недоступна — complete() бросает LLMUnavailableError, пайплайн
продолжает с заголовком по умолчанию. release() выгружает то, что загрузили мы.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from loguru import logger

from llm import lm_studio_api as api
from llm.lm_studio_provider import LLMUnavailableError, LMStudioConfig, LMStudioProvider
from llm.model_selector import (
    DEFAULT_PIPELINE_RESERVE_MIB,
    NoSuitableModelError,
    Selection,
    available_after_unload_mib,
    read_mem_available_mib,
    select_model,
)


class ManagedLMStudio:
    def __init__(
        self,
        config: LMStudioConfig | None = None,
        reserve_mib: float = DEFAULT_PIPELINE_RESERVE_MIB,
        mem_reader=read_mem_available_mib,
    ) -> None:
        self._config = config or LMStudioConfig()
        self._reserve_mib = reserve_mib
        self._mem_reader = mem_reader
        self._lock = threading.Lock()
        self._provider: LMStudioProvider | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self.selection: Selection | None = None
        self._loaded_by_us: list[str] = []

    # ------------------------------------------------------------------ подготовка
    def start_loading_in_background(self) -> None:
        with self._lock:
            if self._thread is None and self._provider is None and self._error is None:
                self._thread = threading.Thread(target=self._prepare, name="lm-studio-prepare", daemon=True)
                self._thread.start()

    def _ensure_ready(self) -> LMStudioProvider:
        if self._provider is None and self._error is None:
            if self._thread is not None:
                self._thread.join()
            else:
                self._prepare()
        if self._provider is None:
            raise LLMUnavailableError(self._error or "LM Studio не подготовлена")
        return self._provider

    def _prepare(self) -> None:
        with self._lock:
            if self._provider is not None:
                return
            try:
                self._provider = self._prepare_unlocked()
            except (api.LMStudioAdminError, NoSuitableModelError) as exc:
                self._error = f"Не удалось подготовить модель LM Studio: {exc}"
                logger.warning("{}", self._error)

    def _prepare_unlocked(self) -> LMStudioProvider:
        base_url = self._config.base_url
        config = self._config

        if not config.manage_models:
            # ручной режим: как раньше — модель задана/выбирается самой LM Studio
            provider = LMStudioProvider(config)
            provider.supports_vision = False
            return provider

        models = api.list_models(base_url)
        available = available_after_unload_mib(models, self._mem_reader())
        selection = select_model(
            models, available_mib=available, override=config.model_override, reserve_mib=self._reserve_mib
        )
        self.selection = selection
        logger.info("LLM: выбрана модель {} — {}", selection.model_key, selection.reason)

        if not selection.already_loaded:
            for model in models:
                for instance in model.loaded_instances:
                    logger.info("LLM: выгружаю {} (освободить память под {})", instance, selection.model_key)
                    api.unload_model(base_url, instance)
            try:
                instance_id = api.load_model(base_url, selection.model_key, context_length=config.context_length)
                self._loaded_by_us.append(instance_id)
                logger.info("LLM: {} загружена (контекст {})", instance_id, config.context_length)
            except api.LMStudioAdminError as exc:
                # LM Studio без REST-загрузки: полагаемся на JIT-загрузку при первом запросе
                logger.warning("LLM: загрузка по REST недоступна ({}) — рассчитываю на JIT-загрузку", exc)

        effort = config.reasoning_effort or selection.reasoning_effort
        provider = LMStudioProvider(replace(config, model_override=selection.model_key, reasoning_effort=effort))
        provider.supports_vision = bool(config.use_vision and selection.supports_vision)
        return provider

    # ------------------------------------------------------------------ LLMProvider
    @property
    def supports_vision(self) -> bool:
        try:
            return self._ensure_ready().supports_vision
        except LLMUnavailableError:
            return False

    @property
    def last_usage(self) -> dict:
        return self._provider.last_usage if self._provider else {}

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, images: list[bytes] | None = None) -> str:
        return self._ensure_ready().complete(system_prompt, user_prompt, max_tokens=max_tokens, images=images)

    def list_models(self) -> list[str]:
        return self._ensure_ready().list_models()

    # ------------------------------------------------------------------ завершение
    def release(self) -> None:
        """Выгружает модели, которые загрузили мы (память нужна пользователю)."""
        with self._lock:
            instances, self._loaded_by_us = self._loaded_by_us, []
            self._provider = None
            self._error = None
            self._thread = None
        for instance in instances:
            try:
                api.unload_model(self._config.base_url, instance)
                logger.info("LLM: {} выгружена", instance)
            except api.LMStudioAdminError as exc:
                logger.warning("LLM: не удалось выгрузить {}: {}", instance, exc)
