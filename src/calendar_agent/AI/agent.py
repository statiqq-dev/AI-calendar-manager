"""Сборка агента: модель + инструменты + системный промпт."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from calendar_agent.AI.llm import build_model
from calendar_agent.AI.tools import TOOLS

SYSTEM_PROMPT = """\
Ты ассистент, который помогает управлять календарём пользователя.
Пока ты умеешь только сообщать текущие дату, время и день недели.

Правила:
- Отвечай по-русски, коротко и по делу.
- Если вопрос связан с датой или временем, сначала узнай текущее время инструментом, не угадывай.
- Сообщай только то, о чём спросили: спросили день недели — назови день недели, а не всё сразу.
"""


def build_agent(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Создаёт агента.

    Модель можно передать снаружи (например, другую или тестовую);
    по умолчанию берётся модель из настроек.
    """
    return create_agent(
        model=model or build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
