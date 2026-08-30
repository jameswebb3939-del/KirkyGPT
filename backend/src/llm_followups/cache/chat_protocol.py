from __future__ import annotations

from typing import Protocol

from ..server.llm_runtime import (
    GenerationResult,
)


class ChatGenerationCache(Protocol):
    """
    Cache abstraction for generated
    LLM responses.
    """

    async def get_generation(
        self,
        request_hash: str,
    ) -> GenerationResult | None:
        ...

    async def set_generation(
        self,
        request_hash: str,
        result: GenerationResult,
    ) -> None:
        ...

    async def ping(self) -> bool:
        ...

    async def close(self) -> None:
        ...