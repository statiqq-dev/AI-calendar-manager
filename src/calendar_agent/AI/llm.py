"""Создание чат-модели.

Единственное место, где проект знает, какая LLM используется.
Модель и провайдер берутся из настроек, поэтому сменить Ollama на Claude
можно правкой .env, без изменений в коде.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from calendar_agent.config import settings


def build_model() -> BaseChatModel:
    """Создаёт объект чат-модели по настройкам из .env."""
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=settings.model_temperature,
        reasoning=False,  # параметр ChatOllama: отключает "размышления" qwen3 (<think>...)
    )
