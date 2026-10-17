"""Календарь в памяти: для разработки и тестов, без настоящего Календаря macOS."""

from datetime import datetime
from uuid import uuid4

from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import EventNotFoundError
from calendar_agent.calendar.models import Event


class FakeCalendarService(CalendarService):
    def __init__(self) -> None:
        self._events: dict[str, Event] = {}

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        # Пересечение: событие началось до конца периода и закончилось после его начала.
        overlapping = [
            event for event in self._events.values()
            if event.start < end and event.end > start
        ]
        return sorted(overlapping, key=lambda event: event.start)

    def create_event(
        self, title: str, start: datetime, end: datetime, notes: str | None = None
    ) -> Event:
        event = Event(id=str(uuid4()), title=title, start=start, end=end, notes=notes)
        self._events[event.id] = event
        return event

    def delete_event(self, event_id: str) -> None:
        if event_id not in self._events:
            raise EventNotFoundError(event_id)
        del self._events[event_id]
