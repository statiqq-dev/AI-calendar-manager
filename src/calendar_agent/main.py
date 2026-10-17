"""Точка входа: задаём агенту вопрос и печатаем всю цепочку сообщений."""

from langchain_core.messages import HumanMessage

from calendar_agent.ai.agent import build_agent

QUESTION = "Какой день недели будет через 10 дней?"


def main() -> None:
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(QUESTION)]})

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
