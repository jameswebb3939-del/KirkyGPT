from __future__ import annotations

from ..cache.protocol import (
    ConversationCache,
)
from ..persistence.models import (
    ConversationModel,
)

from .chat_history import (
    ChatHistoryService,
)


class CachedChatHistoryService:
    """
    Cache-aside decorator for
    ChatHistoryService.

    Writes always go through the
    underlying SQLite-backed service.

    Redis is only an optimisation.
    """

    def __init__(
        self,
        service: ChatHistoryService,
        cache: ConversationCache,
    ) -> None:
        self._service = service
        self._cache = cache

    async def create_conversation(
        self,
        title: str = "New chat",
    ) -> ConversationModel:
        conversation = (
            await self._service
            .create_conversation(
                title=title
            )
        )

        # SQLite commit succeeded before
        # anything is cached.
        await self._cache.set_conversation(
            conversation
        )

        await (
            self._cache
            .invalidate_conversations()
        )

        return conversation

    async def list_conversations(
        self,
    ) -> list[ConversationModel]:
        cached = (
            await self._cache
            .get_conversations()
        )

        if cached is not None:
            return cached

        conversations = (
            await self._service
            .list_conversations()
        )

        await self._cache.set_conversations(
            conversations
        )

        return conversations

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationModel:
        cached = (
            await self._cache
            .get_conversation(
                conversation_id
            )
        )

        if cached is not None:
            return cached

        conversation = (
            await self._service
            .get_conversation(
                conversation_id
            )
        )

        await self._cache.set_conversation(
            conversation
        )

        return conversation

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        # Delete from SQLite first.
        #
        # If the database operation fails,
        # the cache remains untouched.
        await self._service.delete_conversation(
            conversation_id
        )

        await (
            self._cache
            .delete_conversation(
                conversation_id
            )
        )

        await (
            self._cache
            .invalidate_conversations()
        )

    async def send_message(
        self,
        conversation_id: str,
        content: str,
    ) -> ConversationModel:
        # Generation and SQLite transaction
        # remain completely inside the
        # existing service.
        conversation = (
            await self._service.send_message(
                conversation_id,
                content,
            )
        )

        # The database is authoritative.
        # Only refresh cache after the
        # transaction has succeeded.
        await self._cache.set_conversation(
            conversation
        )

        # Title and updated_at may have
        # changed, so list cache is stale.
        await (
            self._cache
            .invalidate_conversations()
        )

        return conversation