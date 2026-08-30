from __future__ import annotations

from typing import Protocol

from ..persistence.models import (
    ConversationModel,
)


class ConversationCache(Protocol):
    """
    Cache abstraction for conversation data.

    SQLite remains the source of truth.
    Implementations must therefore be safe
    to bypass when unavailable.
    """

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationModel | None:
        ...

    async def set_conversation(
        self,
        conversation: ConversationModel,
    ) -> None:
        ...

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        ...

    async def get_conversations(
        self,
    ) -> list[ConversationModel] | None:
        ...

    async def set_conversations(
        self,
        conversations: list[
            ConversationModel
        ],
    ) -> None:
        ...

    async def invalidate_conversations(
        self,
    ) -> None:
        ...

    async def ping(self) -> bool:
        ...

    async def close(self) -> None:
        ...