from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm_followups.eval.core.models import EvalExample
from src.llm_followups.eval.targets.llm_runtime import LLMRuntimeTarget


class FakeRuntime:
    def __init__(self) -> None:
        self.received_messages = None
        self.received_kwargs = None

    def make_request(
        self,
        messages,
        *,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
    ):
        self.received_messages = messages
        self.received_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        return {"request": "fake"}

    async def generate(self, request):
        assert request == {"request": "fake"}

        return SimpleNamespace(
            final_text="- Question one?\n- Question two?\n- Question three?",
            raw_text="raw",
            latency_ms=42,
            used_repair=True,
            used_fallback=False,
        )


@pytest.mark.asyncio
async def test_llm_runtime_target_adapts_runtime_to_target_contract() -> None:
    runtime = FakeRuntime()

    target = LLMRuntimeTarget(
        runtime,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.9,
    )

    prediction = await target.generate(
        EvalExample(
            id=5,
            input="Explain Docker",
        )
    )

    assert prediction.example_id == 5
    assert prediction.output.startswith("- Question one?")
    assert prediction.raw_output == "raw"

    assert prediction.metadata == {
        "latency_ms": 42,
        "used_repair": True,
        "used_fallback": False,
        "target": "llm_runtime",
    }

    assert len(runtime.received_messages) == 1
    assert runtime.received_messages[0].role == "user"
    assert runtime.received_messages[0].content == "Explain Docker"

    assert runtime.received_kwargs == {
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
    }
