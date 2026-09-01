from __future__ import annotations

import pytest
from redis.exceptions import (
    RedisError,
)

from llm_followups.cache.chat_redis import (
    RedisChatGenerationCache,
)
from llm_followups.server.runtime_types import (
    GenerationResult,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[
            str,
            str,
        ] = {}

        self.expiries: dict[
            str,
            int,
        ] = {}

        self.closed = False

    async def ping(self):
        return True

    async def get(
        self,
        key: str,
    ):
        return self.values.get(
            key
        )

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


def make_result():
    return GenerationResult(
        raw_text="raw output",
        final_text=(
            "- Question one?\n"
            "- Question two?\n"
            "- Question three?"
        ),
        used_fallback=False,
        used_repair=True,
        latency_ms=123,
    )


@pytest.mark.asyncio
async def test_generation_round_trip():
    redis = FakeRedis()

    cache = RedisChatGenerationCache(
        url="redis://unused",
        ttl_s=600,
        key_prefix="ec_pro",
        client=redis,
    )

    result = make_result()

    await cache.set_generation(
        "abc123",
        result,
    )

    loaded = (
        await cache.get_generation(
            "abc123"
        )
    )

    assert loaded is not None

    assert (
        loaded.raw_text
        == result.raw_text
    )

    assert (
        loaded.final_text
        == result.final_text
    )

    assert (
        loaded.used_fallback
        is False
    )

    assert (
        loaded.used_repair
        is True
    )

    assert (
        loaded.latency_ms
        == 123
    )


@pytest.mark.asyncio
async def test_generation_ttl():
    redis = FakeRedis()

    cache = RedisChatGenerationCache(
        url="redis://unused",
        ttl_s=600,
        key_prefix="ec_pro",
        client=redis,
    )

    await cache.set_generation(
        "abc123",
        make_result(),
    )

    key = cache.generation_key(
        "abc123"
    )

    assert redis.expiries[key] == 600


@pytest.mark.asyncio
async def test_corrupt_generation_is_miss():
    redis = FakeRedis()

    cache = RedisChatGenerationCache(
        url="redis://unused",
        ttl_s=600,
        key_prefix="ec_pro",
        client=redis,
    )

    key = cache.generation_key(
        "broken"
    )

    redis.values[key] = (
        "not-json"
    )

    result = (
        await cache.get_generation(
            "broken"
        )
    )

    assert result is None

    assert key not in redis.values


@pytest.mark.asyncio
async def test_chat_cache_fails_open():
    redis = BrokenRedis()

    cache = RedisChatGenerationCache(
        url="redis://unused",
        ttl_s=600,
        key_prefix="ec_pro",
        client=redis,
    )

    available = (
        await cache.ping()
    )

    assert available is False

    result = (
        await cache.get_generation(
            "anything"
        )
    )

    assert result is None