"""Точка входа: терминальный чат с агентом.

Единственное место, где создаются конкретные объекты (календарь, хранилище памяти, агент)
и связываются между собой. Остальные модули получают их аргументами.
"""

from typing import Any
from uuid import uuid4

from langchain.agents.middleware.human_in_the_loop import Decision, HITLRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.fake import FakeCalendarService

EXIT_COMMANDS = {"выход", "exit", "quit"}
NEW_CONVERSATION_COMMAND = "/new"
YES_ANSWERS = {"y", "yes", "д", "да"}


def new_conversation() -> RunnableConfig:
    """Config нового разговора: свой thread_id = своя история в checkpointer."""
    return {"configurable": {"thread_id": str(uuid4())}}


def ask_yes_no(question: str) -> bool:
    try:
        return input(f"{question} [y/n] ").strip().lower() in YES_ANSWERS
    except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C на вопросе = «нет»
        print()
        return False


def ask_decisions(request: HITLRequest) -> list[Decision]:
    """Спрашивает человека про каждое действие из паузы, решения в том же порядке."""
    decisions: list[Decision] = []
    for action in request["action_requests"]:
        if ask_yes_no(action["description"]):
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "Пользователь отменил действие."})
    return decisions


def run_turn(agent: CompiledStateGraph, user_input: str, config: RunnableConfig) -> str:
    """Один ход разговора: запрос, подтверждения на паузах, финальный ответ."""
    result: dict[str, Any] = agent.invoke({"messages": [HumanMessage(user_input)]}, config=config)

    while "__interrupt__" in result:  # агент встал на паузу и ждёт решения
        decisions = ask_decisions(result["__interrupt__"][0].value)
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    return result["messages"][-1].content


def main() -> None:
    calendar = FakeCalendarService()
    checkpointer = InMemorySaver()
    agent = build_agent(calendar, checkpointer=checkpointer)
    config = new_conversation()

    print("Календарный ассистент. /new — новый разговор, «выход» или Ctrl+D — выход.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break
        if user_input.lower() == NEW_CONVERSATION_COMMAND:
            config = new_conversation()
            print("Новый разговор.")
            continue

        print(run_turn(agent, user_input, config))


if __name__ == "__main__":
    main()
