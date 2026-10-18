"""Точка входа: терминальный чат с агентом.

Единственное место, где создаются конкретные объекты (календарь, хранилище памяти, агент)
и связываются между собой. Остальные модули получают их аргументами.
"""

import sys
from typing import Any
from uuid import uuid4

from langchain.agents.middleware.human_in_the_loop import Decision, HITLRequest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from calendar_agent.ai.agent import build_agent
from calendar_agent.ai.llm import ModelUnavailableError, check_model
from calendar_agent.calendar.base import CalendarService
from calendar_agent.calendar.exceptions import CalendarError
from calendar_agent.calendar.fake import FakeCalendarService
from calendar_agent.config import settings

try:
    import readline  # noqa: F401  # стрелки, история ввода и нормальное редактирование кириллицы в input()
except ImportError:  # на Windows модуля нет, работаем без него
    pass

EXIT_COMMANDS = {"/exit", "выход", "exit", "quit"}
NEW_CONVERSATION_COMMAND = "/new"
HELP_COMMAND = "/help"
YES_ANSWERS = {"y", "yes", "д", "да"}

# Что показать пользователю, пока агент вызывает tool: видно, что он не завис.
TOOL_PROGRESS = {
    "list_events": "смотрю календарь",
    "find_events": "ищу событие",
    "find_free_time": "ищу свободное время",
    "create_event": "создаю событие",
    "update_event": "меняю событие",
    "delete_event": "готовлю удаление",
}

HELP = """\
Пиши обычными словами, например:
  что у меня завтра?
  поставь созвон с Лёшей в пятницу в 15:00
  найди свободный час в понедельник
  перенеси стоматолога на 17:00
  удали созвон с Лёшей
Команды: /help — подсказка, /new — начать разговор заново, /exit — выход."""


def build_calendar() -> CalendarService:
    """Выбирает реализацию календаря по настройке CALENDAR_BACKEND."""
    if settings.calendar_backend == "macos":
        # Импорт внутри функции: PyObjC есть только на macOS, остальным он не нужен.
        from calendar_agent.calendar.macos import MacCalendarService

        return MacCalendarService(settings.calendar_name)
    return FakeCalendarService()


def calendar_description() -> str:
    if settings.calendar_backend == "macos":
        return f"календарь macOS «{settings.calendar_name}»"
    return "демо-календарь в памяти (события пропадут после выхода, см. README)"


def new_conversation() -> RunnableConfig:
    """Config нового разговора: свой thread_id = своя история в checkpointer."""
    # run_name — имя запуска в трейсах LangSmith, если трейсинг включён.
    return {"configurable": {"thread_id": str(uuid4())}, "run_name": "calendar-agent"}


def ask_yes_no(question: str) -> bool:
    try:
        return input(f"{question} [y/n] ").strip().lower() in YES_ANSWERS
    except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C на вопросе = «нет»
        print()
        return False


def ask_decisions(request: HITLRequest) -> list[Decision]:
    """Спрашивает человека про каждое действие из паузы, решения в том же порядке."""
    decisions: list[Decision] = []
    for action in request["action_requests"]:
        if ask_yes_no(action["description"]):
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "Пользователь отменил действие."})
    return decisions


def show_progress(update: dict[str, Any]) -> None:
    """Печатает, какие tools модель решила вызвать на этом шаге."""
    for message in (update.get("model") or {}).get("messages", []):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"  · {TOOL_PROGRESS.get(call['name'], call['name'])}…")


def run_turn(agent: CompiledStateGraph, user_input: str, config: RunnableConfig) -> str:
    """Один ход разговора: запрос, подтверждения на паузах, финальный ответ.

    stream вместо invoke отдаёт результат каждого шага графа сразу, как только он готов,
    поэтому пользователь видит прогресс, а не ждёт молча.
    """
    payload: Any = {"messages": [HumanMessage(user_input)]}
    while True:
        interrupt = None
        for update in agent.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in update:  # агент встал на паузу и ждёт решения
                interrupt = update["__interrupt__"][0]
            else:
                show_progress(update)
        if interrupt is None:
            break
        payload = Command(resume={"decisions": ask_decisions(interrupt.value)})

    return agent.get_state(config).values["messages"][-1].content


def chat(agent: CompiledStateGraph) -> None:
    """Цикл разговора. Ошибка одного хода не роняет программу."""
    config = new_conversation()
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print()
            return

        command = user_input.lower()
        if not user_input:
            continue
        if command in EXIT_COMMANDS:
            return
        if command == HELP_COMMAND:
            print(HELP)
            continue
        if command == NEW_CONVERSATION_COMMAND:
            config = new_conversation()
            print("Начали заново: прошлый разговор забыт, календарь не изменился.")
            continue

        try:
            print(run_turn(agent, user_input, config))
        except KeyboardInterrupt:
            print("\nОстановил. Можно спросить ещё раз.")
        except ConnectionError:
            print("Потерял связь с моделью. Проверь, что Ollama запущена, и повтори запрос.")
        except ModelCallLimitExceededError:
            print("Не справился за разумное число шагов. Попробуй сформулировать проще или по частям.")
        except Exception as error:  # noqa: BLE001  # последний рубеж: показать причину, не потерять сессию
            print(f"Что-то пошло не так ({type(error).__name__}: {error}). Начинаю разговор заново.")
            config = new_conversation()


def main() -> None:
    try:
        check_model()
        calendar = build_calendar()
    except (ModelUnavailableError, CalendarError) as error:
        sys.exit(f"Не получилось запуститься. {error}")

    agent = build_agent(calendar, checkpointer=InMemorySaver())
    print(f"Календарный ассистент · модель {settings.model_name} · {calendar_description()}\n")
    print(HELP)
    chat(agent)


if __name__ == "__main__":
    main()
