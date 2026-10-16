"""Инструменты агента.

Модель не видит код функций: решение «вызывать или нет» она принимает
только по имени и docstring. Поэтому docstring здесь — это промпт.
"""

from datetime import datetime

from langchain.tools import tool

WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")


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


TOOLS = [get_current_datetime]
