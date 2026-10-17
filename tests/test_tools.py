"""Тесты tools: проверяем без LLM, вызывая tool напрямую через .invoke().

Модель здесь не нужна: tool — это обычный объект, и мы сами подставляем
аргументы так, как их прислала бы модель (даты строками в ISO 8601).
Подготовку данных (Arrange) делаем через календарь напрямую,
а проверяем (Act + Assert) то, что tool вернёт модели.
"""

from datetime import datetime, timedelta, timezone

import pytest
from langchain.tools import BaseTool

from calendar_agent.ai.tools import make_tools
from calendar_agent.calendar.fake import FakeCalendarService

MSK = timezone(timedelta(hours=3))
DAY = {"start": "2026-10-17T00:00:00+03:00", "end": "2026-10-18T00:00:00+03:00"}


def at(hour: int) -> datetime:
    return datetime(2026, 10, 17, hour, tzinfo=MSK)


@pytest.fixture
def calendar() -> FakeCalendarService:
    return FakeCalendarService()


@pytest.fixture
def tools(calendar) -> dict[str, BaseTool]:
    """Tools, привязанные к тому же календарю, что и фикстура calendar."""
    return {t.name: t for t in make_tools(calendar)}


def test_list_empty_period_says_no_events(tools):
    assert tools["list_events"].invoke(DAY) == "В этом периоде событий нет."


def test_list_shows_id_date_and_title(calendar, tools):
    event = calendar.create_event("Созвон с Лёшей", at(15), at(16))

    result = tools["list_events"].invoke(DAY)

    assert result == f"[id={event.id}] 2026-10-17 15:00–16:00 Созвон с Лёшей"


def test_create_adds_event_to_calendar(calendar, tools):
    tools["create_event"].invoke(
        {"title": "Созвон", "start": "2026-10-17T15:00:00+03:00", "end": "2026-10-17T16:00:00+03:00"}
    )

    [event] = calendar.list_events(at(0), at(23))
    assert (event.title, event.start, event.end) == ("Созвон", at(15), at(16))


def test_create_with_end_before_start_returns_error_text(calendar, tools):
    result = tools["create_event"].invoke(
        {"title": "Созвон", "start": "2026-10-17T15:00:00+03:00", "end": "2026-10-17T14:00:00+03:00"}
    )

    assert result.startswith("Ошибка")
    assert calendar.list_events(at(0), at(23)) == []


def test_delete_removes_event(calendar, tools):
    event = calendar.create_event("Созвон", at(15), at(16))

    tools["delete_event"].invoke({"event_id": event.id})

    assert calendar.list_events(at(0), at(23)) == []


def test_delete_unknown_event_returns_error_text(tools):
    result = tools["delete_event"].invoke({"event_id": "нет-такого-id"})

    assert result.startswith("Ошибка")
