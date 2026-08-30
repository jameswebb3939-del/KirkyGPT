from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ..cache.chat_protocol import (
    ChatGenerationCache,
)
from ..server.llm_runtime import (
    GenerationRequest,
    GenerationResult,
    LLMRuntime,
)
from ..server.schemas import (
    ChatMessage,
)
from ..utils.config import (
    Settings,
)


class CachedLLMRuntime:
    """
    Cache-aside decorator around
    LLMRuntime.

    Identical generation requests can
    reuse a Redis-backed GenerationResult.
    """

    def __init__(
        self,
        runtime: LLMRuntime,
        cache: ChatGenerationCache,
        settings: Settings,
    ) -> None:
        self._runtime = runtime
        self._cache = cache
        self._settings = settings

    # ----------------------------------
    # Runtime lifecycle / metadata
    # ----------------------------------

    async def load(self) -> None:
        await self._runtime.load()

    def is_loaded(self) -> bool:
        return self._runtime.is_loaded()

    def model_name(self) -> str:
        return self._runtime.model_name()

    def device_str(self) -> str:
        return self._runtime.device_str()

    def adapter_loaded(self) -> bool:
        return (
            self._runtime
            .adapter_loaded()
        )

    # ----------------------------------
    # Request creation
    # ----------------------------------

    def make_request(
        self,
        messages: Sequence[
            ChatMessage
        ],
        *,
        max_new_tokens: (
            int | None
        ) = None,
        temperature: (
            float | None
        ) = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        return (
            self._runtime
            .make_request(
                messages,
                max_new_tokens=(
                    max_new_tokens
                ),
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        )

    # ----------------------------------
    # Cached generation
    # ----------------------------------

    async def generate(
        self,
        req: GenerationRequest,
    ) -> GenerationResult:
        # Make sure model identity has
        # resolved before constructing
        # the cache key.
        if not self._runtime.is_loaded():
            await self._runtime.load()

        request_hash = (
            self._request_hash(req)
        )

        cached = (
            await self._cache
            .get_generation(
                request_hash
            )
        )

        if cached is not None:
            return cached

        # Do not cache exceptions.
        result = (
            await self._runtime
            .generate(req)
        )

        await self._cache.set_generation(
            request_hash,
            result,
        )

        return result

    # ----------------------------------
    # Key construction
    # ----------------------------------

    def _request_hash(
        self,
        req: GenerationRequest,
    ) -> str:
        payload = {
            "cache_version": 1,

            "model": (
                self._runtime
                .model_name()
            ),

            "adapter_path": (
                str(
                    self._settings
                    .adapter_path
                )
                if (
                    self._settings
                    .adapter_path
                    is not None
                )
                else None
            ),

            "adapter_loaded": (
                self._runtime
                .adapter_loaded()
            ),

            "messages": [
                {
                    "role": (
                        message.role
                    ),
                    "content": (
                        message.content
                    ),
                }
                for message
                in req.messages
            ],

            "max_new_tokens": (
                req.max_new_tokens
            ),

            "temperature": (
                req.temperature
            ),

            "top_p": req.top_p,

            "seed": req.seed,

            "enforce_format": (
                self._settings
                .enforce_format
            ),

            "min_questions": (
                self._settings
                .min_questions
            ),

            "bullet_style": (
                self._settings
                .bullet_style
            ),
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()