"""Тесты подтверждения удаления: агент встаёт на паузу перед delete_event.

Фейковая модель сначала просит удалить событие, потом отвечает текстом.
Проверяем паузу и то, что календарь меняется только после согласия человека.
"""

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.exceptions import EventNotFoundError
from calendar_agent.calendar.fake import FakeCalendarService
from fakes import FakeModel

CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}


@pytest.fixture
def calendar():
    return FakeCalendarService()


@pytest.fixture
def event(calendar):
    return calendar.create_event(
        "Стоматолог", datetime(2026, 10, 20, 14, tzinfo=UTC), datetime(2026, 10, 20, 15, tzinfo=UTC)
    )


def make_agent(calendar, event_id: str):
    model = FakeModel(messages=iter([
        AIMessage("", tool_calls=[
            {"name": "delete_event", "args": {"event_id": event_id}, "id": "call_1"},
        ]),
        AIMessage("Готово."),
    ]))
    return build_agent(calendar, model=model, checkpointer=InMemorySaver())


def ask_to_delete(agent):
    return agent.invoke({"messages": [HumanMessage("Удали стоматолога")]}, config=CONFIG)


def resume(agent, decision: dict):
    return agent.invoke(Command(resume={"decisions": [decision]}), config=CONFIG)


def test_deletion_pauses_before_running(calendar, event):
    agent = make_agent(calendar, event.id)

    result = ask_to_delete(agent)

    [action] = result["__interrupt__"][0].value["action_requests"]
    assert action["description"] == "Удалить «Стоматолог» 20.10.2026 14:00–15:00?"
    assert calendar.get_event_by_id(event.id) == event


def test_rejected_deletion_keeps_event(calendar, event):
    agent = make_agent(calendar, event.id)
    ask_to_delete(agent)

    result = resume(agent, {"type": "reject", "message": "Не надо"})

    assert "__interrupt__" not in result
    assert calendar.get_event_by_id(event.id) == event


def test_approved_deletion_removes_event(calendar, event):
    agent = make_agent(calendar, event.id)
    ask_to_delete(agent)

    resume(agent, {"type": "approve"})

    with pytest.raises(EventNotFoundError):
        calendar.get_event_by_id(event.id)


def test_missing_event_is_not_confirmed(calendar):
    agent = make_agent(calendar, "missing-id")

    result = ask_to_delete(agent)

    assert "__interrupt__" not in result
    assert "не найдено" in result["messages"][-2].content  # ошибку получила модель
