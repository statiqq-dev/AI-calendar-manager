"""Календарь macOS через EventKit (мост PyObjC).

Работает только с одним выделенным календарём (по имени), чтобы агент
не видел и не трогал остальные события пользователя.
Наружу отдаёт только наши Event и CalendarError: объекты EventKit не выходят из модуля.
"""

import threading
from datetime import UTC, datetime
from functools import cache
from typing import Any

from EventKit import EKEntityTypeEvent, EKEvent, EKEventStore, EKSpanThisEvent
from Foundation import NSDate

from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import CalendarError, EventNotFoundError
from calendar_agent.calendar.models import Event

ACCESS_TIMEOUT_SECONDS = 60


class MacCalendarService(CalendarService):
    def __init__(self, calendar_name: str) -> None:
        self._store = _event_store()
        self._calendar = self._find_calendar(calendar_name)

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        predicate = self._store.predicateForEventsWithStartDate_endDate_calendars_(
            _to_nsdate(start), _to_nsdate(end), [self._calendar]
        )
        events = [_to_event(e) for e in self._store.eventsMatchingPredicate_(predicate)]
        # Фильтруем сами, чтобы границы периода работали ровно по контракту CalendarService.
        overlapping = [event for event in events if event.start < end and event.end > start]
        return sorted(overlapping, key=lambda event: event.start)

    def create_event(
        self, title: str, start: datetime, end: datetime, notes: str | None = None
    ) -> Event:
        # Проверяем данные теми же правилами Pydantic, что и фейк, до похода в систему.
        draft = Event(id="new", title=title, start=start, end=end, notes=notes)

        ek_event = EKEvent.eventWithEventStore_(self._store)
        ek_event.setCalendar_(self._calendar)
        ek_event.setTitle_(draft.title)
        ek_event.setStartDate_(_to_nsdate(draft.start))
        ek_event.setEndDate_(_to_nsdate(draft.end))
        ek_event.setNotes_(draft.notes)

        saved, error = self._store.saveEvent_span_commit_error_(ek_event, EKSpanThisEvent, True, None)
        if not saved:
            raise CalendarError(f"Календарь macOS не сохранил событие: {error.localizedDescription()}")
        return _to_event(ek_event)

    def delete_event(self, event_id: str) -> None:
        ek_event = self._find_ek_event(event_id)
        removed, error = self._store.removeEvent_span_commit_error_(ek_event, EKSpanThisEvent, True, None)
        if not removed:
            raise CalendarError(f"Календарь macOS не удалил событие: {error.localizedDescription()}")

    def get_event_by_id(self, event_id: str) -> Event:
        return _to_event(self._find_ek_event(event_id))

    def _find_calendar(self, name: str) -> Any:
        for ek_calendar in self._store.calendarsForEntityType_(EKEntityTypeEvent):
            if ek_calendar.title() == name:
                return ek_calendar
        available = ", ".join(str(c.title()) for c in self._store.calendarsForEntityType_(EKEntityTypeEvent))
        raise CalendarError(
            f"Календарь «{name}» не найден. Создай его в приложении Календарь. Сейчас есть: {available}."
        )

    def _find_ek_event(self, event_id: str) -> Any:
        ek_event = self._store.eventWithIdentifier_(event_id)
        # Событие из чужого календаря считаем несуществующим: агенту виден только свой.
        if ek_event is None or ek_event.calendar().calendarIdentifier() != self._calendar.calendarIdentifier():
            raise EventNotFoundError(event_id)
        return ek_event


@cache
def _event_store() -> Any:
    """Одно подключение к календарям на весь процесс: Apple ограничивает число EKEventStore."""
    store = EKEventStore.alloc().init()
    _request_access(store)
    return store


def _request_access(store: Any) -> None:
    # macOS отвечает асинхронно, в callback из другого потока; ждём ответ через threading.Event.
    answered = threading.Event()
    granted: list[bool] = []
    reasons: list[str] = []

    def on_answer(is_granted: bool, error: Any) -> None:
        granted.append(bool(is_granted))
        if error is not None:
            reasons.append(str(error.localizedDescription()))
        answered.set()

    if store.respondsToSelector_(b"requestFullAccessToEventsWithCompletion:"):  # macOS 14+
        store.requestFullAccessToEventsWithCompletion_(on_answer)
    else:
        store.requestAccessToEntityType_completion_(EKEntityTypeEvent, on_answer)

    if not answered.wait(ACCESS_TIMEOUT_SECONDS) or not granted[0]:
        status = EKEventStore.authorizationStatusForEntityType_(EKEntityTypeEvent)
        reason = reasons[0] if reasons else "без объяснения"
        raise CalendarError(
            f"Нет доступа к Календарю (статус {status}, причина: {reason}). Разреши его в "
            "Системных настройках → Конфиденциальность и безопасность → Календари "
            "для приложения, из которого запускаешь."
        )


def _to_nsdate(moment: datetime) -> Any:
    return NSDate.dateWithTimeIntervalSince1970_(moment.timestamp())


def _from_nsdate(nsdate: Any) -> datetime:
    # NSDate — точка во времени без пояса; показываем её в локальном поясе пользователя.
    return datetime.fromtimestamp(nsdate.timeIntervalSince1970(), tz=UTC).astimezone()


def _to_event(ek_event: Any) -> Event:
    notes = ek_event.notes()
    return Event(
        id=str(ek_event.eventIdentifier()),
        title=str(ek_event.title() or "(без названия)"),
        start=_from_nsdate(ek_event.startDate()),
        end=_from_nsdate(ek_event.endDate()),
        notes=str(notes) if notes else None,
    )
