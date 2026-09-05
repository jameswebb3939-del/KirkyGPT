from __future__ import annotations

import os
import time
from collections.abc import Sequence

from llm_followups.server.runtime_types import GenerationRequest, GenerationResult
from llm_followups.server.schemas import ChatMessage
from llm_followups.utils.config import Settings

from .engine import RuleEngine


def rules_only_enabled() -> bool:
    """
    Use deterministic rules by default.

    Set RULES_ONLY=false later to restore the existing model-backed path.
    """
    return os.getenv("RULES_ONLY", "true").strip().casefold() in {
        "true",
        "1",
        "yes",
        "on",
    }


class RuleRuntime:
    """
    RuntimeProtocol-compatible deterministic runtime.

    It lets the existing FastAPI, SQLite and Redis wiring remain unchanged
    while guaranteeing that generation comes only from predefined rules.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        engine: RuleEngine | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine if engine is not None else RuleEngine()
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def model_name(self) -> str:
        return "kirk-gpt-rules"

    def device_str(self) -> str:
        return "cpu"

    def adapter_loaded(self) -> bool:
        return False

    def make_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        if not messages:
            raise ValueError("At least one message is required")

        return GenerationRequest(
            messages=tuple(messages),
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self._settings.max_new_tokens
            ),
            temperature=(
                temperature
                if temperature is not None
                else self._settings.temperature
            ),
            top_p=(
                top_p
                if top_p is not None
                else self._settings.top_p
            ),
            seed=seed if seed is not None else self._settings.seed,
        )

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        text = self._engine.respond(req.messages)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))

        return GenerationResult(
            raw_text=text,
            final_text=text,
            used_fallback=False,
            used_repair=False,
            latency_ms=latency_ms,
        )
