"""Тесты календаря в памяти.

Каждый тест проверяет одно обещание из контракта CalendarService (см. base.py).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from calendar_agent.calendar.exceptions import EventNotFoundError
from calendar_agent.calendar.fake import FakeCalendarService


def at(hour: int, day: int = 16) -> datetime:
    """Короткий помощник: datetime с часовым поясом на 16 октября 2026."""
    return datetime(2026, 10, day, hour, tzinfo=UTC)


@pytest.fixture
def calendar() -> FakeCalendarService:
    """Каждый тест получает новый пустой календарь."""
    return FakeCalendarService()


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
    with pytest.raises(ValidationError):
        calendar.create_event("Встреча", at(14), at(end_hour))


def test_event_without_timezone_is_rejected(calendar):
    with pytest.raises(ValidationError):
        calendar.create_event("Встреча", datetime(2026, 10, 16, 14), datetime(2026, 10, 16, 15))
