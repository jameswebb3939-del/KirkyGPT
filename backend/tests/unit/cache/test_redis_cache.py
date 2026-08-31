from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import pytest
from redis.exceptions import RedisError

from llm_followups.cache.redis import (
    RedisConversationCache,
)

from llm_followups.persistence.models import (
    ConversationModel,
    MessageModel,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

        self.closed = False

    async def ping(self):
        return True

    async def get(
        self,
        key: str,
    ):
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int,
    ):
        self.values[key] = value
        self.expiries[key] = ex

        return True

    async def delete(
        self,
        key: str,
    ):
        self.values.pop(
            key,
            None,
        )

        self.expiries.pop(
            key,
            None,
        )

        return 1

    async def aclose(self):
        self.closed = True


class BrokenRedis(FakeRedis):
    async def ping(self):
        raise RedisError(
            "redis unavailable"
        )


def make_conversation():
    now = datetime.now(
        timezone.utc
    )

    conversation = ConversationModel(
        id="conversation-1",
        title="Docker",
        created_at=now,
        updated_at=now,
    )

    conversation.messages = [
        MessageModel(
            id="message-1",
            conversation_id=(
                conversation.id
            ),
            role="user",
            content="Explain Docker",
            position=0,
            created_at=now,
        ),
        MessageModel(
            id="message-2",
            conversation_id=(
                conversation.id
            ),
            role="assistant",
            content=(
                "- Question one?\n"
                "- Question two?\n"
                "- Question three?"
            ),
            position=1,
            created_at=now,
        ),
    ]

    return conversation


@pytest.mark.asyncio
async def test_conversation_round_trip():
    redis = FakeRedis()

    cache = RedisConversationCache(
        url="redis://unused",
        ttl_s=300,
        key_prefix="ec_pro",
        client=redis,
    )

    conversation = (
        make_conversation()
    )

    await cache.set_conversation(
        conversation
    )

    loaded = (
        await cache.get_conversation(
            conversation.id
        )
    )

    assert loaded is not None

    assert (
        loaded.id
        == conversation.id
    )

    assert (
        loaded.title
        == "Docker"
    )

    assert len(
        loaded.messages
    ) == 2

    assert [
        message.position
        for message in loaded.messages
    ] == [
        0,
        1,
    ]


@pytest.mark.asyncio
async def test_conversation_uses_ttl():
    redis = FakeRedis()

    cache = RedisConversationCache(
        url="redis://unused",
        ttl_s=123,
        client=redis,
    )

    conversation = (
        make_conversation()
    )

    await cache.set_conversation(
        conversation
    )

    key = (
        cache.conversation_key(
            conversation.id
        )
    )

    assert (
        redis.expiries[key]
        == 123
    )


@pytest.mark.asyncio
async def test_conversation_list_round_trip():
    redis = FakeRedis()

    cache = RedisConversationCache(
        url="redis://unused",
        client=redis,
    )

    conversation = (
        make_conversation()
    )

    await cache.set_conversations(
        [conversation]
    )

    loaded = (
        await cache
        .get_conversations()
    )

    assert loaded is not None
    assert len(loaded) == 1

    assert (
        loaded[0].id
        == conversation.id
    )


@pytest.mark.asyncio
async def test_delete_conversation_cache():
    redis = FakeRedis()

    cache = RedisConversationCache(
        url="redis://unused",
        client=redis,
    )

    conversation = (
        make_conversation()
    )

    await cache.set_conversation(
        conversation
    )

    await cache.delete_conversation(
        conversation.id
    )

    loaded = (
        await cache.get_conversation(
            conversation.id
        )
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_list_invalidation():
    redis = FakeRedis()

    cache = RedisConversationCache(
        url="redis://unused",
        client=redis,
    )

    await cache.set_conversations(
        [make_conversation()]
    )

    assert (
        cache.conversations_key
        in redis.values
    )

    await (
        cache
        .invalidate_conversations()
    )

    assert (
        cache.conversations_key
        not in redis.values
    )


@pytest.mark.asyncio
async def test_redis_failure_is_fail_open():
    cache = RedisConversationCache(
        url="redis://unused",
        client=BrokenRedis(),
    )

    available = await cache.ping()

    assert available is False

    # Cache operations become misses
    # rather than raising.
    result = (
        await cache.get_conversation(
            "anything"
        )
    )

    assert result is None