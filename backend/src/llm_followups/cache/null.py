from __future__ import annotations

from ..persistence.models import (
    ConversationModel,
)


class NullConversationCache:
    """
    No-op conversation cache.

    Used when Redis is disabled or when
    dependency injection requires caching
    to be bypassed.
    """

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationModel | None:
        del conversation_id
        return None

    async def set_conversation(
        self,
        conversation: ConversationModel,
    ) -> None:
        del conversation

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        del conversation_id

    async def get_conversations(
        self,
    ) -> list[ConversationModel] | None:
        return None

    async def set_conversations(
        self,
        conversations: list[
            ConversationModel
        ],
    ) -> None:
        del conversations

    async def invalidate_conversations(
        self,
    ) -> None:
        return None

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None