"""Конфигурация приложения.

Все настройки в одном классе Settings (pydantic-settings).
Значения берутся по приоритету: переменные окружения -> файл .env -> значения по умолчанию.
Использование в коде: `from calendar_agent.config import settings` -> `settings.model_name`.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта: .../AI-agent (config.py -> calendar_agent -> src -> AI-agent).
# Путь к .env считаем от файла, а не от текущей папки: так .env найдётся,
# из какой бы директории ты ни запускал проект.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # лишние переменные в .env не ломают запуск
    )

    # Имена полей = имена переменных в .env (регистр не важен): MODEL_NAME -> model_name
    model_name: str = "qwen3:8b"
    model_provider: str = "ollama"
    # Ключ API: для Ollama не нужен, пригодится для Claude.
    # SecretStr не покажет значение в print/логах; достать: settings.llm_key.get_secret_value()
    llm_key: SecretStr | None = None


# Один объект на всё приложение. Создаётся при первом импорте модуля.
settings = Settings()
