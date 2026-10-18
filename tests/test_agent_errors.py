"""Тесты защитных механизмов агента: ошибки tools, лимит шагов, дата в промпте."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from fakes import FakeModel
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from calendar_agent.ai.agent import build_agent, date_context
from calendar_agent.calendar.fake import FakeCalendarService

CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}


def call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def run(*answers: AIMessage) -> list:
    agent = build_agent(FakeCalendarService(), model=FakeModel(messages=iter(answers)), checkpointer=InMemorySaver())
    return agent.invoke({"messages": [HumanMessage("Сделай что-нибудь")]}, config=CONFIG)["messages"]


def test_calendar_error_goes_back_to_model_as_text():
    messages = run(call("update_event", {"event_id": "missing", "title": "Новое"}), AIMessage("Не нашёл."))

    tool_message = messages[-2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert "не найдено" in tool_message.content
    assert messages[-1].content == "Не нашёл."  # агент не упал и ответил


def test_validation_error_goes_back_to_model_as_text():
    bad_event = {"title": "Созвон", "start": "2026-10-17T15:00:00+03:00", "end": "2026-10-17T14:00:00+03:00"}

    messages = run(call("create_event", bad_event), AIMessage("Исправлю."))

    assert "позже начала" in messages[-2].content


def test_endless_tool_calls_are_stopped():
    endless = (call("list_events", {"start": "2026-10-17T00:00:00Z", "end": "2026-10-18T00:00:00Z"}, f"c{i}")
               for i in range(100))

    with pytest.raises(ModelCallLimitExceededError):
        run(*endless)


def test_date_context_names_today_and_tomorrow():
    now = datetime(2026, 10, 18, 15, 14, tzinfo=timezone(timedelta(hours=3)))

    context = date_context(now)

    assert "Сейчас 2026-10-18 15:14, воскресенье, часовой пояс UTC+03:00." in context
    assert "- 2026-10-19 понедельник (завтра)" in context
    assert "- 2026-10-23 пятница" in context


def test_date_context_handles_utc():
    assert "UTC+00:00" in date_context(datetime(2026, 10, 18, tzinfo=UTC))
