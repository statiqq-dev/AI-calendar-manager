"""Общие подделки для тестов агента."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class FakeModel(GenericFakeChatModel):
    """Модель, которая отвечает заготовленными сообщениями по очереди."""

    def bind_tools(self, tools, **kwargs):  # create_agent прикрепляет tools, фейку они не нужны
        return self
