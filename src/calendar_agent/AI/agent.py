"""Сборка агента: модель + tools + системный промпт."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from calendar_agent.ai.llm import build_model
from calendar_agent.ai.tools import make_tools
from calendar_agent.calendar.base import CalendarService

SYSTEM_PROMPT = """\
Ты ассистент, который управляет календарём пользователя: показывает, создаёт и удаляет события.

Правила работы с датами:
- Перед любыми датами и словами «сегодня», «завтра», «в пятницу» сначала вызови get_current_datetime. Не угадывай дату.
- Даты в инструменты передавай в ISO 8601 с часовым поясом пользователя, например 2026-10-17T15:00:00+03:00.

Правила работы с событиями:
- Чтобы удалить событие, сначала найди его через list_events и возьми id оттуда. Никогда не придумывай id.
- Если инструмент вернул ошибку, исправь аргументы или уточни у пользователя.

Как отвечать:
- По-русски, коротко и по делу. Сообщай только то, о чём спросили.
- Не показывай пользователю id событий.
"""


def build_agent(calendar: CalendarService, model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Создаёт агента, привязанного к календарю.

    Модель можно передать снаружи (например, тестовую);
    по умолчанию берётся модель из настроек.
    """
    return create_agent(
        model=model or build_model(),
        tools=make_tools(calendar),
        system_prompt=SYSTEM_PROMPT,
    )
