from __future__ import annotations

import pytest

from llm_followups.server.llm_runtime import (
    GenerationRequest,
    GenerationResult,
)
from llm_followups.server.schemas import (
    ChatMessage,
)
from llm_followups.services.cached_llm_runtime import (
    CachedLLMRuntime,
)
from llm_followups.utils.config import (
    Settings,
)


class MemoryGenerationCache:
    def __init__(self) -> None:
        self.items = {}

    async def get_generation(
        self,
        request_hash,
    ):
        return self.items.get(
            request_hash
        )

    async def set_generation(
        self,
        request_hash,
        result,
    ):
        self.items[
            request_hash
        ] = result

    async def ping(self):
        return True

    async def close(self):
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.loaded = True

    async def load(self):
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def model_name(self):
        return "fake-model"

    def device_str(self):
        return "cpu"

    def adapter_loaded(self):
        return False

    def make_request(
        self,
        messages,
        *,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
    ):
        return GenerationRequest(
            messages=messages,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens
                is not None
                else 64
            ),
            temperature=(
                temperature
                if temperature
                is not None
                else 0.2
            ),
            top_p=(
                top_p
                if top_p
                is not None
                else 0.9
            ),
            seed=seed,
        )

    async def generate(
        self,
        req,
    ):
        del req

        self.generate_calls += 1

        return GenerationResult(
            raw_text=(
                f"raw-"
                f"{self.generate_calls}"
            ),
            final_text=(
                "- Are you mourning Charlie for the Kirkiversary?\n"
                "- Do you want the Erika timeline?\n"
                "- Should we open with the roof-shot?"
            ),
            used_fallback=False,
            used_repair=False,
            latency_ms=100,
        )


def make_request(
    text: str,
    *,
    max_new_tokens: int = 64,
) -> GenerationRequest:
    return GenerationRequest(
        messages=[
            ChatMessage(
                role="user",
                content=text,
            )
        ],
        max_new_tokens=(
            max_new_tokens
        ),
        temperature=0.2,
        top_p=0.9,
        seed=None,
    )


@pytest.mark.asyncio
async def test_identical_request_hits_cache():
    runtime = FakeRuntime()

    cache = MemoryGenerationCache()

    service = CachedLLMRuntime(
        runtime,  # type: ignore[arg-type]
        cache,    # type: ignore[arg-type]
        Settings(device="cpu"),
    )

    request = make_request(
        "Help me with Kirk"
    )

    first = await service.generate(
        request
    )

    second = await service.generate(
        request
    )

    assert (
        runtime.generate_calls
        == 1
    )

    assert (
        first.final_text
        == second.final_text
    )


@pytest.mark.asyncio
async def test_different_messages_do_not_collide():
    runtime = FakeRuntime()

    cache = MemoryGenerationCache()

    service = CachedLLMRuntime(
        runtime,  # type: ignore[arg-type]
        cache,    # type: ignore[arg-type]
        Settings(device="cpu"),
    )

    await service.generate(
        make_request(
            "Help me with Kirk"
        )
    )

    await service.generate(
        make_request(
            "Help me with conspiracy"
        )
    )

    assert (
        runtime.generate_calls
        == 2
    )


@pytest.mark.asyncio
async def test_generation_parameters_affect_key():
    runtime = FakeRuntime()

    cache = MemoryGenerationCache()

    service = CachedLLMRuntime(
        runtime,  # type: ignore[arg-type]
        cache,    # type: ignore[arg-type]
        Settings(device="cpu"),
    )

    await service.generate(
        make_request(
            "Help me with Kirk",
            max_new_tokens=64,
        )
    )

    await service.generate(
        make_request(
            "Help me with Kirk",
            max_new_tokens=128,
        )
    )

    assert (
        runtime.generate_calls
        == 2
    )


@pytest.mark.asyncio
async def test_full_context_affects_key():
    runtime = FakeRuntime()

    cache = MemoryGenerationCache()

    service = CachedLLMRuntime(
        runtime,  # type: ignore[arg-type]
        cache,    # type: ignore[arg-type]
        Settings(device="cpu"),
    )

    first = GenerationRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Kirk",
            ),
        ],
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        seed=None,
    )

    second = GenerationRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Earlier context",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "- Are you mourning Charlie?"
                ),
            ),
            ChatMessage(
                role="user",
                content="Kirk",
            ),
        ],
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        seed=None,
    )

    await service.generate(first)
    await service.generate(second)

    assert (
        runtime.generate_calls
        == 2
    )