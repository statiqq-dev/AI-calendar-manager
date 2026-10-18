"""Сценарии с настоящей моделью: проверяем поведение агента, а не код.

Это маленькие eval-тесты. Модель недетерминирована, поэтому проверяем
итог в календаре и ключевые признаки ответа, а не точный текст.
Нужна запущенная Ollama, поэтому запускаются только явно:
    RUN_LLM_TESTS=1 uv run pytest tests/test_scenarios_llm.py -v
"""

import os
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.fake import FakeCalendarService
from calendar_agent.cli import new_conversation

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="нужна Ollama: RUN_LLM_TESTS=1")


def tomorrow(hour: int) -> datetime:
    day = datetime.now().astimezone() + timedelta(days=1)
    return day.replace(hour=hour, minute=0, second=0, microsecond=0)


def next_month(calendar: FakeCalendarService) -> list:
    now = datetime.now().astimezone()
    return calendar.list_events(now - timedelta(days=1), now + timedelta(days=30))


def chat(calendar: FakeCalendarService, *messages: str, approve: bool = True) -> str:
    """Проводит разговор из нескольких реплик, на паузах отвечает approve. Возвращает последний ответ."""
    agent = build_agent(calendar, checkpointer=InMemorySaver())
    config = new_conversation()
    for text in messages:
        result = agent.invoke({"messages": [HumanMessage(text)]}, config=config)
        while "__interrupt__" in result:
            actions = result["__interrupt__"][0].value["action_requests"]
            decision = {"type": "approve"} if approve else {"type": "reject", "message": "Пользователь отменил."}
            result = agent.invoke(Command(resume={"decisions": [decision] * len(actions)}), config=config)
    return result["messages"][-1].content


@pytest.fixture
def calendar() -> FakeCalendarService:
    return FakeCalendarService()


@pytest.fixture
def dentist(calendar):
    return calendar.create_event("Стоматолог", tomorrow(14), tomorrow(15))


def test_creates_event_with_default_duration(calendar):
    chat(calendar, "Поставь созвон с Лёшей завтра в 15:00")

    [event] = next_month(calendar)
    assert "лёш" in event.title.lower()
    assert (event.start, event.end) == (tomorrow(15), tomorrow(16))


def test_asks_for_time_instead_of_guessing(calendar):
    answer = chat(calendar, "Запиши меня к стоматологу")

    assert next_month(calendar) == []
    assert "?" in answer


def test_lists_tomorrow(calendar, dentist):
    answer = chat(calendar, "Что у меня завтра?")

    assert "Стоматолог" in answer


def test_moves_event_by_name_and_keeps_duration(calendar, dentist):
    chat(calendar, "Перенеси стоматолога на 17:00")

    moved = calendar.get_event_by_id(dentist.id)
    assert (moved.start, moved.end) == (tomorrow(17), tomorrow(18))


def test_deletes_after_confirmation(calendar, dentist):
    chat(calendar, "Удали стоматолога")

    assert next_month(calendar) == []


def test_rejected_deletion_keeps_event(calendar, dentist):
    chat(calendar, "Удали стоматолога", approve=False)

    assert next_month(calendar) == [dentist]


def test_understands_reference_to_previous_answer(calendar, dentist):
    chat(calendar, "Что у меня завтра?", "Удали его")

    assert next_month(calendar) == []
