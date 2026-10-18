"""Модели данных календаря."""

from typing import Any

from pydantic import AwareDatetime, BaseModel, Field, ValidationError, model_validator

from calendar_agent.calendar.exceptions import InvalidEventError


class Event(BaseModel):
    """Событие в календаре."""

    id: str
    title: str = Field(min_length=1)
    start: AwareDatetime  # только с часовым поясом: наивный datetime не пройдёт валидацию
    end: AwareDatetime
    notes: str | None = None

    @model_validator(mode="after")
    def check_end_after_start(self) -> "Event":
        if self.end <= self.start:
            raise ValueError("Конец события должен быть позже начала")
        return self

    @classmethod
    def checked(cls, **fields: Any) -> "Event":
        """Создаёт событие; ошибки проверки превращает в InvalidEventError с коротким текстом.

        Так слой календаря наружу бросает только свои ошибки (CalendarError),
        а не внутренние ошибки Pydantic.
        """
        try:
            return cls.model_validate(fields)
        except ValidationError as error:
            reasons = "; ".join(err["msg"].removeprefix("Value error, ") for err in error.errors())
            raise InvalidEventError(reasons) from None

    def with_changes(self, **changes: Any) -> "Event":
        """Копия события с изменёнными полями. None значит «не менять», проверки те же."""
        filled = {field: value for field, value in changes.items() if value is not None}
        return Event.checked(**(self.model_dump() | filled))
