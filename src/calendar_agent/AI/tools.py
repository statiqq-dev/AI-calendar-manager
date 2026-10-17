"""Инструменты агента."""

from datetime import datetime

from langchain.tools import BaseTool, tool
from pydantic import ValidationError

from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import CalendarError
from calendar_agent.calendar.models import Event

WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")


def _format_event(event: Event) -> str:
    """Одно событие одной строкой: так модели проще читать список."""
    line = f"[id={event.id}] {event.start:%Y-%m-%d %H:%M}–{event.end:%H:%M} {event.title}"
    if event.notes:
        line += f" ({event.notes})"
    return line


def _validation_message(error: ValidationError) -> str:
    """Короткая причина ошибки валидации без служебного текста Pydantic."""
    return "; ".join(err["msg"] for err in error.errors())


@tool
def get_current_datetime() -> str:
    """Возвращает текущие дату, время, день недели и часовой пояс пользователя.

    Вызывай перед любым ответом, где нужна сегодняшняя дата или текущее время,
    а также перед вычислениями со словами «сегодня», «завтра», «через N дней».
    Пример ответа: "2026-10-16 15:14 (UTC+03:00), пятница".
    """
    now = datetime.now().astimezone()
    offset = now.strftime("%z")  # "+0300"
    return f"{now:%Y-%m-%d %H:%M} (UTC{offset[:3]}:{offset[3:]}), {WEEKDAYS[now.weekday()]}"


def make_tools(calendar: CalendarService) -> list[BaseTool]:
    """Собирает все tools агента, привязанные к переданному календарю.

    Календарь попадает в tools через замыкание: модель передаёт только
    аргументы из JSON, а с каким календарём работать, решает тот, кто вызывает фабрику.
    """

    @tool(parse_docstring=True)
    def list_events(start: datetime, end: datetime) -> str:
        """Показывает события, которые полностью или частично попадают в период.

        Используй, чтобы ответить «что у меня в календаре» и чтобы найти id события
        перед удалением. Каждое событие в ответе — отдельная строка с его id.

        Args:
            start: Начало периода в ISO 8601 с часовым поясом, например 2026-10-17T00:00:00+03:00.
            end: Конец периода в том же формате, например 2026-10-18T00:00:00+03:00.
        """
        events = calendar.list_events(start, end)
        if not events:
            return "В этом периоде событий нет."
        return "\n".join(_format_event(event) for event in events)

    @tool(parse_docstring=True)
    def create_event(title: str, start: datetime, end: datetime, notes: str | None = None) -> str:
        """Создаёт новое событие в календаре.

        Если пользователь не назвал длительность, считай, что событие длится 1 час.

        Args:
            title: Короткое название, например «Созвон с Лёшей».
            start: Начало в ISO 8601 с часовым поясом, например 2026-10-17T15:00:00+03:00.
            end: Конец в том же формате, строго позже начала.
            notes: Необязательная заметка к событию.
        """
        try:
            event = calendar.create_event(title=title, start=start, end=end, notes=notes)
        except ValidationError as error:
            return f"Ошибка: событие не создано. {_validation_message(error)}"
        except CalendarError as error:
            return f"Ошибка: событие не создано. {error}"
        return f"Событие создано: {_format_event(event)}"

    @tool(parse_docstring=True)
    def delete_event(event_id: str) -> str:
        """Удаляет событие из календаря по его id.

        Id бери только из ответа list_events, никогда не придумывай его.

        Args:
            event_id: Id события из ответа list_events.
        """
        try:
            calendar.delete_event(event_id)
        except CalendarError as error:
            return f"Ошибка: {error}. Найди событие через list_events."
        return f"Событие {event_id} удалено."

    return [get_current_datetime, list_events, create_event, delete_event]
