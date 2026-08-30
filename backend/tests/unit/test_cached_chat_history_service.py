from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import pytest

from llm_followups.services.cached_chat_history import (
    CachedChatHistoryService,
)

from llm_followups.persistence.models import (
    ConversationModel,
)


class FakeService:
    def __init__(
        self,
        conversation,
    ) -> None:
        self.conversation = conversation
        self.get_calls = 0

    async def get_conversation(
        self,
        conversation_id: str,
    ):
        self.get_calls += 1

        assert (
            conversation_id
            == self.conversation.id
        )

        return self.conversation


class MemoryCache:
    def __init__(self) -> None:
        self.items = {}

    async def get_conversation(
        self,
        conversation_id,
    ):
        return self.items.get(
            conversation_id
        )

    async def set_conversation(
        self,
        conversation,
    ):
        self.items[
            conversation.id
        ] = conversation

    async def delete_conversation(
        self,
        conversation_id,
    ):
        self.items.pop(
            conversation_id,
            None,
        )

    async def get_conversations(self):
        return None

    async def set_conversations(
        self,
        conversations,
    ):
        del conversations

    async def invalidate_conversations(
        self,
    ):
        return None

    async def ping(self):
        return True

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_cache_hit_avoids_service():
    now = datetime.now(
        timezone.utc
    )

    conversation = ConversationModel(
        id="c1",
        title="Docker",
        created_at=now,
        updated_at=now,
    )

    conversation.messages = []

    underlying = FakeService(
        conversation
    )

    cache = MemoryCache()

    service = (
        CachedChatHistoryService(
            underlying,  # type: ignore[arg-type]
            cache,       # type: ignore[arg-type]
        )
    )

    first = (
        await service.get_conversation(
            "c1"
        )
    )

    second = (
        await service.get_conversation(
            "c1"
        )
    )

    assert first.id == "c1"
    assert second.id == "c1"

    # First request = SQLite/service.
    # Second request = cache.
    assert underlying.get_calls == 1