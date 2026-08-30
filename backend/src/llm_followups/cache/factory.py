from __future__ import annotations

from .chat_null import (
    NullChatGenerationCache,
)
from .chat_protocol import (
    ChatGenerationCache,
)
from .chat_redis import (
    RedisChatGenerationCache,
)
from .null import (
    NullConversationCache,
)
from .protocol import (
    ConversationCache,
)
from .redis import (
    RedisConversationCache,
)

from ..utils.config import (
    Settings,
)


def build_conversation_cache(
    settings: Settings,
) -> ConversationCache:
    if not settings.redis_enabled:
        return NullConversationCache()

    return RedisConversationCache(
        url=settings.redis_url,
        ttl_s=(
            settings
            .redis_cache_ttl_s
        ),
        key_prefix=(
            settings
            .redis_key_prefix
        ),
    )


def build_chat_generation_cache(
    settings: Settings,
) -> ChatGenerationCache:
    if (
        not settings.redis_enabled
        or not (
            settings
            .redis_chat_cache_enabled
        )
    ):
        return (
            NullChatGenerationCache()
        )

    return RedisChatGenerationCache(
        url=settings.redis_url,
        ttl_s=(
            settings
            .redis_chat_cache_ttl_s
        ),
        key_prefix=(
            settings
            .redis_key_prefix
        ),
    )