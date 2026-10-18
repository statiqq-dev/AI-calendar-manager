"""Создание чат-модели.

Единственное место, где проект знает, какая LLM используется.
Модель и провайдер берутся из настроек, поэтому сменить Ollama на Claude
можно правкой .env и установкой интеграции провайдера, без изменений в коде.
"""

from typing import Any

import ollama
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from calendar_agent.config import settings


class ModelUnavailableError(Exception):
    """Модель нельзя использовать: сервер не запущен или модель не скачана."""


def build_model() -> BaseChatModel:
    """Создаёт объект чат-модели по настройкам из .env."""
    extra: dict[str, Any] = {}
    if settings.model_provider == "ollama":
        extra["reasoning"] = False  # параметр ChatOllama: отключает "размышления" qwen3 (<think>...)
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=settings.model_temperature,
        **extra,
    )


def check_model() -> None:
    """Проверяет до начала разговора, что Ollama запущена и модель скачана.

    Без проверки пользователь узнал бы об этом только трейсбеком после первого вопроса.
    """
    if settings.model_provider != "ollama":
        return
    try:
        ollama.show(settings.model_name)
    except ConnectionError:
        raise ModelUnavailableError(
            "Ollama не запущена. Открой приложение Ollama или выполни `ollama serve`."
        ) from None
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise ModelUnavailableError(
                f"Модель {settings.model_name} не скачана. Выполни `ollama pull {settings.model_name}`."
            ) from None
        raise
