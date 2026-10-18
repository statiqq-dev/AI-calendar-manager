"""Сборка агента: модель, tools, системный промпт и middleware.

Middleware по порядку:
1. current_date_prompt — каждый вызов модели получает свежую дату и ближайшие дни;
2. ToolErrorMiddleware — ошибки календаря (CalendarError) возвращаются модели текстом, чтобы она исправилась;
3. HumanInTheLoopMiddleware — пауза и подтверждение человеком перед удалением;
4. ModelCallLimitMiddleware — предохранитель от зацикливания на одном запросе.
"""

from datetime import datetime, timedelta
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallRequest,
    ToolErrorMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from calendar_agent.ai.llm import build_model
from calendar_agent.ai.tools import make_tools
from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import CalendarError, EventNotFoundError

MAX_MODEL_CALLS_PER_TURN = 12
DAYS_AHEAD = 14
WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")

SYSTEM_PROMPT = """\
Ты ассистент, который ведёт календарь пользователя. Ты умеешь показывать события, \
искать событие по названию, искать свободное время, создавать, переносить, \
переименовывать и удалять события. На другие темы вежливо скажи, что умеешь только это.

Даты и время:
- Сегодняшняя дата и ближайшие дни указаны в конце. Считай «завтра», «в пятницу», \
«на следующей неделе» только по этому списку.
- В инструменты передавай даты в ISO 8601 с часовым поясом пользователя, например 2026-10-20T15:00:00+03:00.
- «Утром» — 09:00, «днём» — 13:00, «вечером» — 19:00, если точное время не названо.
- Время с 1 до 7 без слов «утра» или «ночи» считай дневным: «в 4» — это 16:00.
- Время без даты — ближайшее такое время в будущем.
- «Сегодня», «завтра» — весь день с 00:00 до 00:00 следующего дня. «На неделе» — с сегодня до конца воскресенья.

Создание события:
- Нужны название и время начала. Если не сказано, в какой день или во сколько, \
задай один короткий вопрос и ничего не создавай.
- Название бери из слов пользователя: «созвон с Лёшей», «стоматолог». На «поставь встречу» назови «Встреча».
- Если длительность не названа, не передавай end: инструмент сам поставит 1 час.

Изменение и удаление:
- Если событие названо словами, найди его через find_events. Если оно упоминалось \
в разговоре («её», «вторую»), возьми id из прошлых ответов инструментов. Никогда не придумывай id.
- Если подходит несколько событий, перечисли их и спроси, какое изменить. Если ни одного, так и скажи.
- Перенос и переименование делай через update_event. При переносе без новой длительности передавай только start.
- Удаление подтверждает пользователь, система спросит его сама. Сразу вызывай delete_event, \
не переспрашивай. Если пользователь отказался, не повторяй удаление.

Ошибки и пересечения:
- Если инструмент вернул «Ошибка», исправь аргументы и попробуй ещё раз или объясни пользователю, что не так.
- Если инструмент предупредил о пересечении с другими событиями, обязательно скажи об этом.

Как отвечать:
- По-русски, коротко, обычным текстом без таблиц. Не показывай id событий.
- После изменения календаря назови итог: название, день недели, дату и время, \
например «Создал «Созвон с Лёшей»: вт 20.10, 15:00–16:00».
- Если что-то выбрал сам (длительность 1 час, «вечером» = 19:00), коротко скажи об этом.
"""


def build_agent(
    calendar: CalendarService,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Собирает агента. model и checkpointer можно подменить в тестах."""
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        current_date_prompt,
        ToolErrorMiddleware(tool_error_message),
        HumanInTheLoopMiddleware(interrupt_on={"delete_event": confirm_deletion(calendar)}),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS_PER_TURN, exit_behavior="error"),
    ]
    return create_agent(
        model=model or build_model(),
        tools=make_tools(calendar),
        checkpointer=checkpointer,
        middleware=middleware,
    )


def date_context(now: datetime) -> str:
    """Текущая дата и календарь на две недели вперёд: маленькой модели не придётся считать дни."""
    offset = now.strftime("%z")  # "+0300"
    lines = [
        f"Сейчас {now:%Y-%m-%d %H:%M}, {WEEKDAYS[now.weekday()]}, часовой пояс UTC{offset[:3]}:{offset[3:]}.",
        "Ближайшие дни:",
    ]
    labels = {0: " (сегодня)", 1: " (завтра)", 2: " (послезавтра)"}
    for shift in range(DAYS_AHEAD):
        day = now + timedelta(days=shift)
        lines.append(f"- {day:%Y-%m-%d} {WEEKDAYS[day.weekday()]}{labels.get(shift, '')}")
    return "\n".join(lines)


@dynamic_prompt
def current_date_prompt(request: ModelRequest) -> str:
    """Системный промпт собирается перед каждым вызовом модели, поэтому дата всегда свежая."""
    return f"{SYSTEM_PROMPT}\n{date_context(datetime.now().astimezone())}"


def tool_error_message(error: Exception, request: ToolCallRequest) -> str | None:
    """Ожидаемые ошибки отдаём модели текстом. Остальные — баги: пусть падают громко."""
    if isinstance(error, EventNotFoundError):
        return f"Ошибка: {error}. Найди событие через find_events или list_events."
    if isinstance(error, CalendarError):
        return f"Ошибка: {error}."
    return None


def confirm_deletion(calendar: CalendarService) -> InterruptOnConfig:
    """Настройки паузы перед delete_event: когда спрашивать и какой вопрос показать."""

    def event_exists(request: ToolCallRequest) -> bool:
        # Несуществующее событие не подтверждаем: tool вернёт модели ошибку.
        try:
            calendar.get_event_by_id(request.tool_call["args"].get("event_id", ""))
        except EventNotFoundError:
            return False
        return True

    def describe(tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[Any]) -> str:
        event = calendar.get_event_by_id(tool_call["args"]["event_id"])
        return f"Удалить «{event.title}» {event.start:%d.%m.%Y %H:%M}–{event.end:%H:%M}?"

    return InterruptOnConfig(allowed_decisions=["approve", "reject"], description=describe, when=event_exists)
