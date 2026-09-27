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
Модель не ответила за таймаут (LLMTimeoutError) — она понижается в рейтинге (llm/model_health.py),
причина уходит в интерфейс, а модель выбирается заново в фоне (следующее видео очереди пойдёт с ней).
"""

from __future__ import annotations

import threading
from dataclasses import replace

from loguru import logger

from core.entities.llm_issue import LLMIssue
from llm import lm_studio_api as api
from llm.lm_studio_provider import LLMTimeoutError, LLMUnavailableError, LMStudioConfig, LMStudioProvider
from llm.model_health import ModelHealth
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
        on_issue=None,
        health: ModelHealth | None = None,
    ) -> None:
        self._config = config or LMStudioConfig()
        self._health = health or ModelHealth()     # таймауты моделей (понижение в рейтинге)
        self._reserve_mib = reserve_mib
        self._mem_reader = mem_reader
        self._lock = threading.Lock()              # короткий: только состояние
        self._lifecycle = threading.Lock()         # долгий: загрузка/выгрузка моделей не идут одновременно
        self._users = 0                            # сколько пакетов/видео сейчас пользуются моделью
        self._provider: LMStudioProvider | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self.selection: Selection | None = None
        self._loaded_by_us: list[str] = []
        self.issue: LLMIssue | None = None      # почему модели нет (None — всё в порядке или ещё не пробовали)
        self._on_issue = on_issue               # колбэк(LLMIssue | None): из фонового потока — слушатель сам переходит в UI-поток

    def set_reserve_mib(self, reserve_mib: float) -> None:
        """Новый запас под пайплайн из настроек; действует при следующем выборе модели."""
        self._reserve_mib = reserve_mib

    def _set_issue(self, issue: LLMIssue | None) -> None:
        self.issue = issue
        if self._on_issue is not None:
            try:
                self._on_issue(issue)
            except Exception as exc:   # слушатель не должен ломать подготовку модели
                logger.warning("LLM: слушатель причины отказал: {}", exc)

    def _issue_from_error(self, exc: Exception) -> LLMIssue:
        if isinstance(exc, NoSuitableModelError):
            if exc.need_mib is not None:
                need, free, reserve = exc.need_mib / 1024, (exc.available_mib or 0) / 1024, (exc.reserve_mib or 0) / 1024
                return LLMIssue(
                    "no_memory",
                    f"LLM не загружена: самой лёгкой подходящей модели ({exc.lightest_key}) нужно ~{need:.1f} ГиБ, "
                    f"свободно {free:.1f} ГиБ при запасе {reserve:.1f} ГиБ под пайплайн.",
                    "Закройте лишние программы и контейнеры (например ollama, open-webui), уменьшите «Запас памяти под "
                    "пайплайн» в настройках или скачайте в LM Studio небольшую модель — она будет использована как запасная. "
                    "Пока клипы получают заголовки по умолчанию.",
                    need_gib=need, free_gib=free, reserve_gib=reserve,
                )
            return LLMIssue("no_models", "В LM Studio нет чат-моделей (только embedding).",
                            "Скачайте LLM во вкладке Discover в LM Studio. Пока клипы получают заголовки по умолчанию.")
        return LLMIssue(
            "unavailable", f"LM Studio недоступна на {self._config.base_url} ({exc}).",
            "Запустите LM Studio и включите локальный сервер (Developer → Start Server), затем обработайте видео снова. "
            "Пока клипы получают заголовки по умолчанию.",
        )

    # ------------------------------------------------------------------ учёт пользователей
    def begin_use(self) -> None:
        """Очередь начала работу: модель нужна, пока не вызван парный end_use(). Загрузка стартует в фоне."""
        with self._lock:
            self._users += 1
        self.start_loading_in_background()

    def end_use(self) -> bool:
        """Очередь закончила. True — пользователей не осталось (можно звать release_if_unused())."""
        with self._lock:
            self._users = max(0, self._users - 1)
            return self._users == 0

    def release_if_unused(self) -> bool:
        """Выгружает модель, только если пользователей нет (и они не появились, пока ждали блокировку).
        Блокирующий — вызывать из фонового потока."""
        with self._lifecycle:
            with self._lock:
                if self._users > 0:
                    return False
            self._release_locked()
        return True

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
        with self._lifecycle:
            if self._provider is not None:
                return
            try:
                provider = self._prepare_unlocked()
                with self._lock:
                    self._provider = provider
                if self.issue is None or self.issue.kind != "timeout":   # о таймауте пользователь должен узнать и после перевыбора
                    self._set_issue(None)
            except (api.LMStudioAdminError, NoSuitableModelError) as exc:
                with self._lock:
                    self._error = f"Не удалось подготовить модель LM Studio: {exc}"
                logger.warning("{}", self._error)
                self._set_issue(self._issue_from_error(exc))

    def _prepare_unlocked(self) -> LMStudioProvider:
        base_url = self._config.base_url
        config = self._config

        if not config.manage_models:
            # ручной режим: как раньше — модель задана/выбирается самой LM Studio
            provider = LMStudioProvider(config)
            provider.supports_vision = False
            return provider

        models = api.list_models(base_url)
        mem_available = self._mem_reader()
        available = available_after_unload_mib(models, mem_available)
        loaded = [i for m in models for i in m.loaded_instances]
        if loaded:
            logger.info(
                "LLM: в LM Studio уже загружены {}; свободно {:.1f} ГиБ, после их выгрузки ~{:.1f} ГиБ",
                ", ".join(loaded), mem_available / 1024, available / 1024,
            )
        selection = select_model(
            models, available_mib=available, override=config.model_override, reserve_mib=self._reserve_mib,
            penalized=self._health.penalized(),
        )
        self.selection = selection
        logger.info("LLM: выбрана модель {} — {}", selection.model_key, selection.reason)

        if not selection.already_loaded:
            for model in models:
                for instance in model.loaded_instances:
                    logger.info("LLM: выгружаю {} (освободить память под {})", instance, selection.model_key)
                    api.unload_model(base_url, instance)
                    with self._lock:
                        if instance in self._loaded_by_us:
                            self._loaded_by_us.remove(instance)
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

    def _call(self, request):
        provider = self._ensure_ready()
        try:
            result = request(provider)
        except LLMTimeoutError as exc:
            self._on_timeout(provider, exc)
            raise
        self._health.record_success(provider.last_model)
        return result

    def _on_timeout(self, provider: LMStudioProvider, exc: LLMTimeoutError) -> None:
        """Модель зависла или слишком медленная: понижаем её, сообщаем и в фоне выбираем модель заново."""
        self._health.record_failure(exc.model, f"не ответила за {exc.timeout_sec:.0f} с")
        manual = bool(self._config.model_override)
        self._set_issue(LLMIssue(
            "timeout",
            f"Модель {exc.model} не ответила за {exc.timeout_sec:.0f} с (зависла или слишком медленная для этой машины)"
            + ("." if manual else " и понижена в рейтинге на 3 дня."),
            "Клипы этого видео получают заголовки по умолчанию; "
            + ("модель задана вручную (LM_STUDIO_MODEL_OVERRIDE в .env) — смените её или увеличьте LM_STUDIO_REQUEST_TIMEOUT_SEC."
               if manual else
               "следующее видео пойдёт с заново выбранной моделью. Можно увеличить LM_STUDIO_REQUEST_TIMEOUT_SEC в .env."),
        ))
        if manual:
            return   # перевыбирать не из чего
        with self._lock:
            if self._provider is not provider:
                return   # уже перевыбрана другим потоком
            self._provider = None
            self._thread = None
            self._error = None
        self.start_loading_in_background()

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, images: list[bytes] | None = None) -> str:
        return self._call(lambda p: p.complete(system_prompt, user_prompt, max_tokens=max_tokens, images=images))

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        max_tokens: int = 2048,
        images: list[bytes] | None = None,
        schema_name: str = "response",
    ) -> str:
        return self._call(lambda p: p.complete_json(
            system_prompt, user_prompt, schema, max_tokens=max_tokens, images=images, schema_name=schema_name
        ))

    def list_models(self) -> list[str]:
        return self._ensure_ready().list_models()

    # ------------------------------------------------------------------ завершение
    def release(self) -> None:
        """Безусловно выгружает модели, которые загрузили мы (память нужна пользователю)."""
        with self._lifecycle:
            self._release_locked()

    def _release_locked(self) -> None:
        with self._lock:
            instances, self._loaded_by_us = self._loaded_by_us, []
            self._provider = None
            self._error = None
            self._thread = None
        self.issue = None
        for instance in instances:
            try:
                api.unload_model(self._config.base_url, instance)
                logger.info("LLM: {} выгружена", instance)
            except api.LMStudioAdminError as exc:
                logger.warning("LLM: не удалось выгрузить {}: {}", instance, exc)
