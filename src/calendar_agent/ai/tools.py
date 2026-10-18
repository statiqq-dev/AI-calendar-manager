"""Инструменты агента: тонкие обёртки над CalendarService, удобные для модели.

Всё, что можно посчитать кодом (длительность по умолчанию, свободные окна,
пересечения), считает код, а не модель: так поведение предсказуемо.
Ошибки tools не ловят: CalendarError превращает в текст для модели
middleware в agent.py.
"""

from datetime import datetime, timedelta
from typing import Annotated

from langchain.tools import BaseTool, tool
from pydantic import Field

from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.models import Event

DEFAULT_DURATION = timedelta(hours=1)
SEARCH_DAYS = 90  # насколько вперёд ищет find_events, если период не задан
WEEKDAYS_SHORT = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def format_event(event: Event) -> str:
    """Одно событие одной строкой: id для модели, день недели, чтобы не ошибалась с датами."""
    end = f"{event.end:%H:%M}" if event.end.date() == event.start.date() else f"{event.end:%Y-%m-%d %H:%M}"
    line = f"[id={event.id}] {event.start:%Y-%m-%d} ({WEEKDAYS_SHORT[event.start.weekday()]}) "
    line += f"{event.start:%H:%M}–{end} {event.title}"
    if event.notes:
        line += f" ({event.notes})"
    return line


def format_events(events: list[Event], empty: str) -> str:
    return "\n".join(format_event(event) for event in events) if events else empty


def matches(event: Event, query: str) -> bool:
    """Нестрогое совпадение: каждое слово запроса без окончания есть в названии или заметке.

    «стоматолога» найдёт «Стоматолог», «Лёшей» найдёт «Созвон с Лёшей».
    """
    text = _normalize(f"{event.title} {event.notes or ''}")
    words = [word for word in _normalize(query).split() if len(word) > 2]  # без предлогов «у», «с»
    return all(_stem(word) in text for word in words)


def free_slots(
    busy: list[Event], start: datetime, end: datetime, duration: timedelta
) -> list[tuple[datetime, datetime]]:
    """Промежутки внутри [start, end) без событий, не короче duration. busy отсортирован по start."""
    slots = []
    cursor = start
    for event in busy:
        if event.start - cursor >= duration:
            slots.append((cursor, event.start))
        cursor = max(cursor, event.end)
    if end - cursor >= duration:
        slots.append((cursor, end))
    return slots


def make_tools(calendar: CalendarService) -> list[BaseTool]:
    """Собирает tools агента, привязанные к переданному календарю (через замыкание)."""

    def with_conflicts(event: Event) -> str:
        """Описание события плюс предупреждение, если оно пересекается с другими."""
        others = [e for e in calendar.list_events(event.start, event.end) if e.id != event.id]
        result = format_event(event)
        if others:
            result += "\nВнимание, пересекается с:\n" + format_events(others, empty="")
        return result

    @tool(parse_docstring=True)
    def list_events(start: datetime, end: datetime) -> str:
        """Показывает все события, которые хотя бы частично попадают в период.

        Используй для вопросов «что у меня завтра», «какие планы на неделе».

        Args:
            start: Начало периода в ISO 8601 с часовым поясом, например 2026-10-17T00:00:00+03:00.
            end: Конец периода в том же формате, например 2026-10-18T00:00:00+03:00.
        """
        return format_events(calendar.list_events(start, end), empty="В этом периоде событий нет.")

    @tool(parse_docstring=True)
    def find_events(
        query: Annotated[str, Field(min_length=1)],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> str:
        """Ищет события по словам из названия или заметки.

        Используй, когда пользователь называет событие словами («стоматолог», «созвон с Лёшей»),
        чтобы узнать его id перед изменением или удалением.
        Без периода ищет с начала сегодняшнего дня на 90 дней вперёд.

        Args:
            query: Одно-два ключевых слова, например «стоматолог» или «Лёша».
            start: Необязательное начало периода поиска в ISO 8601 с часовым поясом.
            end: Необязательный конец периода поиска в том же формате.
        """
        today = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        start = start or today
        end = end or start + timedelta(days=SEARCH_DAYS)
        found = [event for event in calendar.list_events(start, end) if matches(event, query)]
        return format_events(found, empty=f"Событий по запросу «{query}» не найдено.")

    @tool(parse_docstring=True)
    def find_free_time(start: datetime, end: datetime, duration_minutes: Annotated[int, Field(gt=0)] = 60) -> str:
        """Ищет свободные промежутки в периоде, куда помещается событие нужной длины.

        Если пользователь не назвал часы, ищи в рабочее время дня: с 09:00 до 21:00.

        Args:
            start: Начало периода в ISO 8601 с часовым поясом.
            end: Конец периода в том же формате.
            duration_minutes: Сколько минут нужно, по умолчанию 60.
        """
        busy = calendar.list_events(start, end)
        slots = free_slots(busy, start, end, timedelta(minutes=duration_minutes))
        if not slots:
            return "Свободного времени нужной длины в этом периоде нет."
        return "\n".join(
            f"{s:%Y-%m-%d} ({WEEKDAYS_SHORT[s.weekday()]}) {s:%H:%M}–{e:%H:%M}" for s, e in slots
        )

    @tool(parse_docstring=True)
    def create_event(title: str, start: datetime, end: datetime | None = None, notes: str | None = None) -> str:
        """Создаёт новое событие.

        Если пользователь не назвал длительность или время окончания, не передавай end.

        Args:
            title: Короткое название из слов пользователя, например «Созвон с Лёшей».
            start: Начало в ISO 8601 с часовым поясом, например 2026-10-17T15:00:00+03:00.
            end: Конец в том же формате. Если не передан, событие длится 1 час.
            notes: Необязательная заметка.
        """
        event = calendar.create_event(title=title, start=start, end=end or start + DEFAULT_DURATION, notes=notes)
        result = f"Создано: {with_conflicts(event)}"
        if end is None:
            result += "\nДлительность не была указана, поставлен 1 час."
        return result

    @tool(parse_docstring=True)
    def update_event(
        event_id: str,
        title: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        notes: str | None = None,
    ) -> str:
        """Меняет существующее событие: переносит, переименовывает или меняет заметку.

        Передавай только то, что меняется. Если передать только start,
        событие переедет целиком и сохранит длительность.

        Args:
            event_id: Id события из ответа list_events или find_events.
            title: Новое название.
            start: Новое начало в ISO 8601 с часовым поясом.
            end: Новый конец в том же формате.
            notes: Новая заметка.
        """
        if start is not None and end is None:
            current = calendar.get_event_by_id(event_id)
            end = start + (current.end - current.start)
        event = calendar.update_event(event_id, title=title, start=start, end=end, notes=notes)
        return f"Изменено: {with_conflicts(event)}"

    @tool(parse_docstring=True)
    def delete_event(event_id: str) -> str:
        """Удаляет событие. Перед удалением система сама спросит подтверждение у пользователя.

        Args:
            event_id: Id события из ответа list_events или find_events.
        """
        calendar.delete_event(event_id)
        return "Событие удалено."

    return [list_events, find_events, find_free_time, create_event, update_event, delete_event]


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _stem(word: str) -> str:
    """Грубо отрезает окончание: русские слова меняются в конце («Лёша», «Лёшей»)."""
    return word[: max(3, len(word) - 2)]
