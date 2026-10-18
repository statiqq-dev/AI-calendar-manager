"""Конфигурация приложения.

Все настройки в одном классе Settings (pydantic-settings).
Значения берутся по приоритету: переменные окружения -> файл .env -> значения по умолчанию.
Использование в коде: `from calendar_agent.config import settings` -> `settings.model_name`.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта (config.py -> calendar_agent -> src -> корень).
# Путь к .env считаем от файла, а не от текущей папки: так .env найдётся,
# из какой бы директории ты ни запускал проект.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Settings читает .env сам, но сторонние библиотеки (например LangSmith с LANGSMITH_*)
# смотрят только в переменные окружения. Поэтому кладём .env ещё и в окружение процесса.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # лишние переменные в .env не ломают запуск
    )

    # Имена полей = имена переменных в .env (регистр не важен): MODEL_NAME -> model_name
    model_name: str = "qwen3:8b"
    # Ключ облачного провайдера пишется в .env его стандартной переменной (например
    # ANTHROPIC_API_KEY): её читает сама интеграция LangChain, отдельное поле не нужно.
    model_provider: str = "ollama"
    # 0 = максимально предсказуемые ответы, то что нужно агенту
    model_temperature: float = 0.0

    # Какой календарь использовать: "fake" — в памяти, "macos" — Календарь macOS через EventKit.
    calendar_backend: Literal["fake", "macos"] = "fake"
    # Агент работает только с этим календарём macOS; создай его в приложении Календарь.
    calendar_name: str = "AI Agent"


# Один объект на всё приложение. Создаётся при первом импорте модуля.
settings = Settings()
