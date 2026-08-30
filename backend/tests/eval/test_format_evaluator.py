from __future__ import annotations

import pytest

from llm_followups.eval.core.models import EvalExample, EvalPrediction
from llm_followups.eval.evaluators.format import (
    FollowupFormatEvaluator,
    RawFollowupFormatEvaluator,
)


VALID = (
    "- What environment are you using?\n"
    "- What are you trying to deploy?\n"
    "- Are you using Docker Compose?"
)


@pytest.mark.asyncio
async def test_valid_final_followup_format_passes() -> None:
    result = await FollowupFormatEvaluator(
        min_questions=3,
        bullet_style="dash",
    ).evaluate(
        EvalExample(id=1, input="Help me with Docker"),
        EvalPrediction(example_id=1, output=VALID),
    )

    assert result.evaluator == "followup_format"
    assert result.passed is True
    assert result.score is True
    assert result.reason is None
    assert result.metadata["num_questions"] == 3


@pytest.mark.asyncio
async def test_raw_format_is_evaluated_before_guardrails() -> None:
    prediction = EvalPrediction(
        example_id=1,
        output=VALID,
        raw_output=VALID + "\nExtra prose",
    )

    final_result = await FollowupFormatEvaluator().evaluate(
        EvalExample(id=1, input="Docker"), prediction
    )
    raw_result = await RawFollowupFormatEvaluator().evaluate(
        EvalExample(id=1, input="Docker"), prediction
    )

    assert final_result.passed is True
    assert raw_result.evaluator == "raw_followup_format"
    assert raw_result.passed is False
    assert raw_result.reason is not None


@pytest.mark.asyncio
async def test_missing_raw_output_fails_raw_evaluator() -> None:
    result = await RawFollowupFormatEvaluator().evaluate(
        EvalExample(id=1, input="Docker"),
        EvalPrediction(example_id=1, output=VALID, raw_output=None),
    )
    assert result.passed is False
    assert "did not contain" in (result.reason or "")
