"""Точка входа: терминальный чат с агентом.

Единственное место, где создаются конкретные объекты (календарь, агент)
и связываются между собой. Остальные модули получают их аргументами.
"""

from langchain_core.messages import HumanMessage

from calendar_agent.ai.agent import build_agent
from calendar_agent.calendar.fake import FakeCalendarService

EXIT_COMMANDS = {"выход", "exit", "quit"}


def main() -> None:
    calendar = FakeCalendarService()
    agent = build_agent(calendar)

    print("Календарный ассистент. Для выхода напиши «выход» или нажми Ctrl+D.")
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

        result = agent.invoke({"messages": [HumanMessage(user_input)]})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
