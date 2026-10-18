"""Тесты одного хода CLI: прогресс, подтверждение удаления, финальный ответ."""

from datetime import UTC, datetime

import pytest
from fakes import FakeModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.exceptions import EventNotFoundError
from calendar_agent.calendar.fake import FakeCalendarService
from calendar_agent.cli import new_conversation, run_turn


@pytest.fixture
def calendar():
    return FakeCalendarService()


@pytest.fixture
def event(calendar):
    return calendar.create_event(
        "Стоматолог", datetime(2026, 10, 20, 14, tzinfo=UTC), datetime(2026, 10, 20, 15, tzinfo=UTC)
    )


def delete_then_answer(calendar, event_id: str):
    model = FakeModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "delete_event", "args": {"event_id": event_id}, "id": "call_1"}]),
        AIMessage("Удалил."),
    ]))
    return build_agent(calendar, model=model, checkpointer=InMemorySaver())


def test_turn_asks_confirmation_and_returns_answer(calendar, event, monkeypatch, capsys):
    questions = []
    monkeypatch.setattr("builtins.input", lambda question: questions.append(question) or "да")

    answer = run_turn(delete_then_answer(calendar, event.id), "Удали стоматолога", new_conversation())

    assert answer == "Удалил."
    assert questions == ["Удалить «Стоматолог» 20.10.2026 14:00–15:00? [y/n] "]
    assert "готовлю удаление" in capsys.readouterr().out
    with pytest.raises(EventNotFoundError):
        calendar.get_event_by_id(event.id)


def test_turn_with_refusal_keeps_event(calendar, event, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda question: "n")

    run_turn(delete_then_answer(calendar, event.id), "Удали стоматолога", new_conversation())

    assert calendar.get_event_by_id(event.id) == event
