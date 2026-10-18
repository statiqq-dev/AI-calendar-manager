"""Интерфейс календаря: что умеет любой календарь, без деталей реализации."""

from abc import ABC, abstractmethod
from datetime import datetime

from calendar_agent.calendar.models import Event


class CalendarService(ABC):
    @abstractmethod
    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        """Возвращает события, пересекающиеся с периодом [start, end), по возрастанию start.

        Событие попадает в период, если хотя бы частично в него заходит.
        Событие, которое только касается границы (закончилось ровно в start
        или началось ровно в end), не попадает.
        """

    @abstractmethod
    def create_event(
        self, title: str, start: datetime, end: datetime, notes: str | None = None
    ) -> Event:
        """Создаёт событие и возвращает его с присвоенным id.

        Если данные неверные (конец не позже начала, нет часового пояса), бросает InvalidEventError.
        """

    @abstractmethod
    def update_event(
        self,
        event_id: str,
        *,
        title: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        notes: str | None = None,
    ) -> Event:
        """Меняет переданные поля события (None — оставить как есть) и возвращает его.

        Если события нет, бросает EventNotFoundError, если данные неверные — InvalidEventError.
        """

    @abstractmethod
    def delete_event(self, event_id: str) -> None:
        """Удаляет событие по id. Если его нет, бросает EventNotFoundError."""

    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Event:
        """Возвращает событие по id. Если его нет, бросает EventNotFoundError."""
