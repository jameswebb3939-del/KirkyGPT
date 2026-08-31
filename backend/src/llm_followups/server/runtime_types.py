from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llm_followups.server.schemas import (
    ChatMessage,
)


@dataclass(frozen=True)
class GenerationRequest:
    messages: Sequence[ChatMessage]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    final_text: str
    used_fallback: bool
    used_repair: bool
    latency_ms: int