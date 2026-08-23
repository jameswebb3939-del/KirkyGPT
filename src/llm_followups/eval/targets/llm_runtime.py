from __future__ import annotations

from llm_followups.eval.core.models import EvalExample, EvalPrediction
from llm_followups.server.llm_runtime import LLMRuntime
from llm_followups.server.schemas import ChatMessage


class LLMRuntimeTarget:
    """Adapter exposing the existing LLMRuntime through the generic Target contract."""

    def __init__(self, runtime: LLMRuntime, *, max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> None:
        self._runtime = runtime
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

    async def generate(self, example: EvalExample) -> EvalPrediction:
        message = ChatMessage(role="user", content=example.input)

        request = self._runtime.make_request(
            [message],
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        result = await self._runtime.generate(request)

        return EvalPrediction(
            example_id=example.id,
            output=result.final_text,
            raw_output=result.raw_text,
            metadata={
                "latency_ms": result.latency_ms,
                "used_repair": result.used_repair,
                "used_fallback": result.used_fallback,
                "target": "llm_runtime",
            },
        )
