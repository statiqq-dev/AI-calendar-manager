"""Этап 0: подключение к модели и первый вызов."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from calendar_agent.config import settings


def build_model() -> BaseChatModel:
    """Создаёт объект чат-модели."""
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=0,     # 0 = максимально предсказуемые ответы, то что нужно агенту
        reasoning=False,   # параметр ChatOllama: отключает "размышления" qwen3 (<think>...)
    )


def main() -> None:
    model = build_model()

    messages = [
        SystemMessage("Ты модель, которая всегда мяукает"),
        HumanMessage("Привет"),
    ]

    response: AIMessage = model.invoke(messages)

    print("CONTENT:", response.content)
    print("USAGE:  ", response.usage_metadata)
    print("RAW:    ", repr(response))


if __name__ == "__main__":
    main()
