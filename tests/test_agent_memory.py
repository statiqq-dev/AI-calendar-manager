"""Тесты памяти агента: checkpointer хранит историю по thread_id.

Настоящая LLM здесь не нужна: фейковая модель отвечает заготовленными репликами,
а проверяем мы то, что сохранилось в состоянии агента.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.fake import FakeCalendarService
from fakes import FakeModel


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def make_agent(*answers: str):
    model = FakeModel(messages=iter([AIMessage(answer) for answer in answers]))
    return build_agent(FakeCalendarService(), model=model, checkpointer=InMemorySaver())


def history(agent, config: RunnableConfig) -> list[str]:
    return [m.content for m in agent.get_state(config).values["messages"]]


def test_same_thread_keeps_history():
    agent = make_agent("Ответ 1", "Ответ 2")

    agent.invoke({"messages": [HumanMessage("Вопрос 1")]}, config=thread("a"))
    agent.invoke({"messages": [HumanMessage("Вопрос 2")]}, config=thread("a"))

    assert history(agent, thread("a")) == ["Вопрос 1", "Ответ 1", "Вопрос 2", "Ответ 2"]


def test_other_thread_starts_empty():
    agent = make_agent("Ответ 1", "Ответ 2")

    agent.invoke({"messages": [HumanMessage("Вопрос 1")]}, config=thread("a"))
    agent.invoke({"messages": [HumanMessage("Вопрос 2")]}, config=thread("b"))

    assert history(agent, thread("b")) == ["Вопрос 2", "Ответ 2"]
