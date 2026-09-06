from __future__ import annotations

import pytest

from llm_followups.rules.runtime import RuleRuntime
from llm_followups.server.schemas import ChatMessage
from llm_followups.utils.config import Settings


@pytest.mark.asyncio
async def test_rule_runtime_needs_no_model() -> None:
    runtime = RuleRuntime(Settings(device="cpu"))

    assert runtime.is_loaded() is False

    await runtime.load()

    assert runtime.is_loaded() is True
    assert runtime.model_name() == "kirk-gpt-rules"
    assert runtime.adapter_loaded() is False

    request = runtime.make_request(
        [
            ChatMessage(
                role="user",
                content="Help me with conspiracy",
            )
        ]
    )

    result = await runtime.generate(request)

    assert result.final_text == (
        "Are you mapping the cover-up, the beneficiaries, or the next target?"
    )
    assert result.raw_text == result.final_text
    assert result.used_fallback is False
    assert result.used_repair is False