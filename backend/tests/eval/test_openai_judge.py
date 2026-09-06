from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_followups.eval.judges.openai import OpenAIJudge


class FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self._content
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self, content: str | None) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(content)
        )


@pytest.mark.asyncio
async def test_openai_judge_normalizes_valid_json_response() -> None:
    client = FakeClient(
        '{"score": 3, "reason": "Very clear"}'
    )

    judge = OpenAIJudge(
        client=client,
        model="test-judge-model",
    )

    result = await judge.judge(
        instructions="Score coherence from 0 to 3.",
        input_text="Help me with Kirk",
        output_text="Are you mourning Charlie for the Kirkiversary?",
    )

    assert result.score == 3.0
    assert result.reason == "Very clear"
    assert result.metadata == {
        "judge_provider": "openai",
        "judge_model": "test-judge-model",
    }

    calls = client.chat.completions.calls
    assert len(calls) == 1

    call = calls[0]
    assert call["model"] == "test-judge-model"
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}

    prompt = call["messages"][1]["content"]
    assert "Score coherence from 0 to 3." in prompt
    assert "Help me with Kirk" in prompt
    assert "Are you mourning Charlie for the Kirkiversary?" in prompt


@pytest.mark.asyncio
async def test_openai_judge_rejects_non_numeric_score() -> None:
    client = FakeClient(
        '{"score": "3", "reason": "wrong type"}'
    )

    judge = OpenAIJudge(client=client)

    with pytest.raises(
        ValueError,
        match="invalid score",
    ):
        await judge.judge(
            instructions="rubric",
            input_text="input",
            output_text="output",
        )


@pytest.mark.asyncio
async def test_openai_judge_rejects_empty_response() -> None:
    judge = OpenAIJudge(
        client=FakeClient(None),
    )

    with pytest.raises(
        ValueError,
        match="empty response",
    ):
        await judge.judge(
            instructions="rubric",
            input_text="input",
            output_text="output",
        )


@pytest.mark.asyncio
async def test_openai_judge_rejects_invalid_json() -> None:
    judge = OpenAIJudge(
        client=FakeClient("not-json"),
    )

    with pytest.raises(json.JSONDecodeError):
        await judge.judge(
            instructions="rubric",
            input_text="input",
            output_text="output",
        )