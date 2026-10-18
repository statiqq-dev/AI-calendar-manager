"""Ошибки слоя календаря."""


class CalendarError(Exception):
    """Базовая ошибка модуля календаря."""


class EventNotFoundError(CalendarError):
    """Событие отсутствует в календаре."""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        super().__init__(f"Событие {event_id} не найдено")


class InvalidEventError(CalendarError):
    """Данные события не прошли проверку: например, конец раньше начала."""
