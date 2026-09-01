from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from llm_followups.server.runtime_types import (
    GenerationRequest,
    GenerationResult,
)
from llm_followups.server.schemas import (
    ChatMessage,
)


class RuntimeProtocol(Protocol):
    """
    Application-facing LLM runtime
    contract.

    Implemented by local Transformers,
    remote inference, and test runtimes.
    """

    async def load(self) -> None:
        ...

    def is_loaded(self) -> bool:
        ...

    def model_name(self) -> str:
        ...

    def device_str(self) -> str:
        ...

    def adapter_loaded(self) -> bool:
        ...

    def make_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        ...

    async def generate(
        self,
        req: GenerationRequest,
    ) -> GenerationResult:
        ...
