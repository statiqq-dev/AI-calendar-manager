"""Сборка агента: модель + tools + системный промпт + подтверждение удаления."""

from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ToolCallRequest,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from calendar_agent.ai.llm import build_model
from calendar_agent.ai.tools import make_tools
from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import EventNotFoundError

SYSTEM_PROMPT = """\
Ты ассистент, который управляет календарём пользователя: показывает, создаёт и удаляет события.

Правила работы с датами:
- Перед любыми датами и словами «сегодня», «завтра», «в пятницу» сначала вызови get_current_datetime. Не угадывай дату.
- Даты в инструменты передавай в ISO 8601 с часовым поясом пользователя, например 2026-10-17T15:00:00+03:00.

Правила работы с событиями:
- Чтобы удалить событие, сначала найди его через list_events и возьми id оттуда. Никогда не придумывай id.
- Удаление подтверждает пользователь, система спросит его сама. Не переспрашивай, сразу вызывай delete_event.
- Если пользователь отклонил удаление, не повторяй его.
- Если инструмент вернул ошибку, исправь аргументы или уточни у пользователя.

Как отвечать:
- По-русски, коротко и по делу. Сообщай только то, о чём спросили.
- Не показывай пользователю id событий.
"""


def build_agent(
    calendar: CalendarService,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Создаёт агента, привязанного к календарю.

    Модель можно передать снаружи (например, тестовую);
    по умолчанию берётся модель из настроек.
    checkpointer — хранилище истории разговоров; без него агент ничего не помнит
    и не может встать на паузу для подтверждения.
    """
    return create_agent(
        model=model or build_model(),
        tools=make_tools(calendar),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"delete_event": confirm_deletion(calendar)},
            ),
        ],
    )


def confirm_deletion(calendar: CalendarService) -> InterruptOnConfig:
    """Политика для delete_event: спросить человека, если удалять есть что."""

    def event_exists(request: ToolCallRequest) -> bool:
        # Несуществующий id не спрашиваем: delete_event вернёт модели ошибку, и она исправится.
        try:
            calendar.get_event_by_id(request.tool_call["args"].get("event_id", ""))
        except EventNotFoundError:
            return False
        return True

    def describe(tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[Any]) -> str:
        # Вызывается только если event_exists вернул True, поэтому событие точно есть.
        event = calendar.get_event_by_id(tool_call["args"]["event_id"])
        return f"Удалить «{event.title}» {event.start:%d.%m.%Y %H:%M}–{event.end:%H:%M}?"

    return InterruptOnConfig(
        allowed_decisions=["approve", "reject"],
        description=describe,
        when=event_exists,
    )
