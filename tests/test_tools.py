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
from calendar_agent.calendar.exceptions import EventNotFoundError
from calendar_agent.calendar.fake import FakeCalendarService

MSK = timezone(timedelta(hours=3))
DAY = {"start": "2026-10-17T00:00:00+03:00", "end": "2026-10-18T00:00:00+03:00"}


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 17, hour, minute, tzinfo=MSK)


def iso(hour: int) -> str:
    return at(hour).isoformat()


@pytest.fixture
def calendar() -> FakeCalendarService:
    return FakeCalendarService()


@pytest.fixture
def tools(calendar) -> dict[str, BaseTool]:
    """Tools, привязанные к тому же календарю, что и фикстура calendar."""
    return {t.name: t for t in make_tools(calendar)}


# --- list_events ---

def test_list_empty_period_says_no_events(tools):
    assert tools["list_events"].invoke(DAY) == "В этом периоде событий нет."


def test_list_shows_id_weekday_time_and_title(calendar, tools):
    event = calendar.create_event("Созвон с Лёшей", at(15), at(16))

    result = tools["list_events"].invoke(DAY)

    assert result == f"[id={event.id}] 2026-10-17 (сб) 15:00–16:00 Созвон с Лёшей"


# --- find_events ---

@pytest.mark.parametrize("query", ["стоматолога", "Стоматолог", "у стоматолога"])
def test_find_ignores_case_and_word_endings(calendar, tools, query):
    calendar.create_event("Стоматолог", at(15), at(16))

    result = tools["find_events"].invoke({"query": query, **DAY})

    assert "Стоматолог" in result


def test_find_treats_yo_as_e(calendar, tools):
    calendar.create_event("Созвон с Лёшей", at(15), at(16))

    assert "Лёшей" in tools["find_events"].invoke({"query": "леша", **DAY})


def test_find_skips_other_events(calendar, tools):
    calendar.create_event("Стоматолог", at(15), at(16))

    assert tools["find_events"].invoke({"query": "созвон", **DAY}) == "Событий по запросу «созвон» не найдено."


# --- find_free_time ---

def test_free_time_between_events(calendar, tools):
    calendar.create_event("Утро", at(9), at(10))
    calendar.create_event("Обед", at(13), at(14))

    result = tools["find_free_time"].invoke({"start": iso(9), "end": iso(15), "duration_minutes": 60})

    assert result.splitlines() == ["2026-10-17 (сб) 10:00–13:00", "2026-10-17 (сб) 14:00–15:00"]


def test_free_time_skips_too_short_gaps(calendar, tools):
    calendar.create_event("Утро", at(9), at(10))
    calendar.create_event("Обед", at(10, 30), at(15))

    result = tools["find_free_time"].invoke({"start": iso(9), "end": iso(15), "duration_minutes": 60})

    assert result == "Свободного времени нужной длины в этом периоде нет."


# --- create_event ---

def test_create_without_end_lasts_one_hour(calendar, tools):
    result = tools["create_event"].invoke({"title": "Созвон", "start": iso(15)})

    [event] = calendar.list_events(at(0), at(23))
    assert (event.title, event.start, event.end) == ("Созвон", at(15), at(16))
    assert "поставлен 1 час" in result


def test_create_warns_about_overlap(calendar, tools):
    calendar.create_event("Стоматолог", at(15), at(16))

    result = tools["create_event"].invoke({"title": "Созвон", "start": iso(15), "end": iso(17)})

    assert "пересекается" in result
    assert "Стоматолог" in result


# --- update_event ---

def test_move_keeps_duration(calendar, tools):
    event = calendar.create_event("Созвон", at(15), at(16, 30))

    tools["update_event"].invoke({"event_id": event.id, "start": iso(18)})

    moved = calendar.get_event_by_id(event.id)
    assert (moved.start, moved.end) == (at(18), at(19, 30))


def test_rename_keeps_time(calendar, tools):
    event = calendar.create_event("Созвон", at(15), at(16))

    tools["update_event"].invoke({"event_id": event.id, "title": "Созвон с Лёшей"})

    renamed = calendar.get_event_by_id(event.id)
    assert (renamed.title, renamed.start, renamed.end) == ("Созвон с Лёшей", at(15), at(16))


# --- delete_event ---

def test_delete_removes_event(calendar, tools):
    event = calendar.create_event("Созвон", at(15), at(16))

    tools["delete_event"].invoke({"event_id": event.id})

    assert calendar.list_events(at(0), at(23)) == []


def test_tool_errors_are_raised_not_hidden(tools):
    # Текст для модели из ошибки делает middleware агента (см. test_agent_errors.py).
    with pytest.raises(EventNotFoundError):
        tools["delete_event"].invoke({"event_id": "нет-такого-id"})
