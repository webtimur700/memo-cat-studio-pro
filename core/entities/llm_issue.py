"""Почему LLM не сработала: причина и что с этим можно сделать (для очереди и карточки клипа).

Раньше это было видно только в логе: пайплайн молча ставил заголовок «Момент N s». Теперь причина
хранится структурно, пишется в JSON клипа и показывается пользователю.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMIssue:
    kind: str                       # no_memory | unavailable | no_models | request_failed | partial
    message: str                    # что случилось и с цифрами (нужно / свободно)
    hint: str                       # что можно сделать
    need_gib: float | None = None
    free_gib: float | None = None
    reserve_gib: float | None = None

    @property
    def text(self) -> str:
        return f"{self.message} {self.hint}".strip()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "message": self.message, "hint": self.hint,
            "need_gib": self.need_gib, "free_gib": self.free_gib, "reserve_gib": self.reserve_gib,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "LLMIssue | None":
        if not data or not data.get("message"):
            return None
        return cls(
            kind=str(data.get("kind") or "unavailable"), message=str(data["message"]), hint=str(data.get("hint") or ""),
            need_gib=data.get("need_gib"), free_gib=data.get("free_gib"), reserve_gib=data.get("reserve_gib"),
        )

    @classmethod
    def from_errors(cls, errors: list[str] | tuple[str, ...], partial: bool = False) -> "LLMIssue":
        """Запасной вариант, когда структурной причины нет: собираем из текстов ошибок запросов."""
        joined = "; ".join(errors)
        if partial:
            return cls("partial", f"Часть данных от LLM не получена ({joined}).",
                       "Обработайте видео ещё раз или проверьте модель в LM Studio.")
        return cls("request_failed", f"LLM не ответила ({joined}).",
                   "Проверьте, что LM Studio запущена, сервер включён и модель загружена, и обработайте видео снова.")
