"""Контрактные тесты CalendarService: одни и те же тесты для каждой реализации.

Каждый тест проверяет одно обещание из контракта (см. base.py).
macOS-вариант трогает настоящий Календарь, поэтому запускается только явно:
    RUN_MACOS_TESTS=1 uv run pytest
и только в отдельном пустом календаре TEST_CALENDAR_NAME.
"""

import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import EventNotFoundError, InvalidEventError
from calendar_agent.calendar.fake import FakeCalendarService

TEST_CALENDAR_NAME = "AI Agent Tests"


def at(hour: int, day: int = 16) -> datetime:
    """Короткий помощник: datetime с часовым поясом на 16 октября 2026."""
    return datetime(2026, 10, day, hour, tzinfo=UTC)


@pytest.fixture(params=["fake", "macos"])
def calendar(request) -> Iterator[CalendarService]:
    """Каждый тест получает пустой календарь: сначала фейк, потом macOS."""
    if request.param == "fake":
        yield FakeCalendarService()
    else:
        yield from macos_calendar()


def macos_calendar() -> Iterator[CalendarService]:
    if sys.platform != "darwin" or os.environ.get("RUN_MACOS_TESTS") != "1":
        pytest.skip("macOS-календарь: запусти на macOS с RUN_MACOS_TESTS=1")
    from calendar_agent.calendar.macos import MacCalendarService

    service = MacCalendarService(TEST_CALENDAR_NAME)
    clear(service)  # вдруг остались события от упавшего прогона
    yield service
    clear(service)  # тест не оставляет мусора в настоящем календаре


def clear(service: CalendarService) -> None:
    for event in service.list_events(at(0, day=1), at(0, day=31)):
        service.delete_event(event.id)


def test_created_event_is_listed(calendar):
    created = calendar.create_event("Встреча", at(14), at(15))

    assert created.id
    assert calendar.list_events(at(0), at(23)) == [created]


def test_event_outside_period_is_not_listed(calendar):
    calendar.create_event("Встреча", at(14), at(15))

    assert calendar.list_events(at(17), at(18)) == []


def test_event_partially_in_period_is_listed(calendar):
    created = calendar.create_event("Встреча", at(14), at(16))

    assert calendar.list_events(at(15), at(18)) == [created]


def test_event_touching_period_border_is_not_listed(calendar):
    # Встреча 14:00–15:00 закончилась ровно в начале периода 15:00–16:00 и не занимает его.
    calendar.create_event("Встреча", at(14), at(15))

    assert calendar.list_events(at(15), at(16)) == []


def test_events_are_sorted_by_start(calendar):
    later = calendar.create_event("Позже", at(14), at(15))
    earlier = calendar.create_event("Раньше", at(10), at(12))

    assert calendar.list_events(at(9), at(16)) == [earlier, later]


def test_deleted_event_is_not_listed(calendar):
    created = calendar.create_event("Встреча", at(14), at(15))

    calendar.delete_event(created.id)

    assert calendar.list_events(at(0), at(23)) == []


def test_delete_unknown_event_raises(calendar):
    with pytest.raises(EventNotFoundError):
        calendar.delete_event("нет-такого-id")


@pytest.mark.parametrize("end_hour", [13, 14], ids=["end-before-start", "end-equals-start"])
def test_event_must_end_after_start(calendar, end_hour):
    with pytest.raises(InvalidEventError):
        calendar.create_event("Встреча", at(14), at(end_hour))


def test_event_without_timezone_is_rejected(calendar):
    with pytest.raises(InvalidEventError):
        calendar.create_event("Встреча", datetime(2026, 10, 16, 14), datetime(2026, 10, 16, 15))


# --- Изменение ---

def test_updated_fields_change_and_others_stay(calendar):
    created = calendar.create_event("Встреча", at(14), at(15), notes="Офис")

    updated = calendar.update_event(created.id, title="Созвон", start=at(16), end=at(17))

    assert (updated.id, updated.title, updated.start, updated.end, updated.notes) == (
        created.id, "Созвон", at(16), at(17), "Офис"
    )
    assert calendar.get_event_by_id(created.id) == updated


def test_update_with_end_before_start_raises_and_keeps_event(calendar):
    created = calendar.create_event("Встреча", at(14), at(15))

    with pytest.raises(InvalidEventError):
        calendar.update_event(created.id, start=at(16))

    assert calendar.get_event_by_id(created.id) == created


def test_update_unknown_event_raises(calendar):
    with pytest.raises(EventNotFoundError):
        calendar.update_event("нет-такого-id", title="Созвон")


# --- Поиск по id ---

def test_get_event_by_id_returns_event(calendar):
    created = calendar.create_event("Встреча", at(14), at(15))

    assert calendar.get_event_by_id(created.id) == created


def test_get_event_by_id_missing_raises(calendar):
    with pytest.raises(EventNotFoundError):
        calendar.get_event_by_id("missing-id")
