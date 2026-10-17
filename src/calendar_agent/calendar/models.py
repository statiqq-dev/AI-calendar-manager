"""Модели данных календаря."""

from pydantic import AwareDatetime, BaseModel, Field, model_validator


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
