from __future__ import annotations

from ..server.llm_runtime import (
    GenerationResult,
)


class NullChatGenerationCache:
    """
    No-op generation cache.

    Used when Redis generation caching
    is disabled or bypassed in tests.
    """

    async def get_generation(
        self,
        request_hash: str,
    ) -> GenerationResult | None:
        del request_hash

        return None

    async def set_generation(
        self,
        request_hash: str,
        result: GenerationResult,
    ) -> None:
        del request_hash
        del result

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None