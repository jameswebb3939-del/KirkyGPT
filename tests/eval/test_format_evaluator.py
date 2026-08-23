from __future__ import annotations

import pytest

from src.llm_followups.eval.core.models import EvalExample, EvalPrediction
from src.llm_followups.eval.evaluators.format import FollowupFormatEvaluator


@pytest.mark.asyncio
async def test_valid_followup_format_passes() -> None:
    evaluator = FollowupFormatEvaluator(
        min_questions=3,
        bullet_style="dash",
    )

    example = EvalExample(
        id=1,
        input="Help me with Docker",
    )

    prediction = EvalPrediction(
        example_id=1,
        output=(
            "- What environment are you using?\n"
            "- What are you trying to deploy?\n"
            "- Are you using Docker Compose?"
        ),
    )

    result = await evaluator.evaluate(example, prediction)

    assert result.evaluator == "followup_format"
    assert result.passed is True
    assert result.score is True
    assert result.reason is None
    assert result.metadata["num_questions"] == 3
    assert result.metadata["normalized_text"] == prediction.output


@pytest.mark.asyncio
async def test_invalid_followup_format_fails() -> None:
    evaluator = FollowupFormatEvaluator()

    example = EvalExample(
        id=1,
        input="Help me with Docker",
    )

    prediction = EvalPrediction(
        example_id=1,
        output="Here are some questions for you.",
    )

    result = await evaluator.evaluate(example, prediction)

    assert result.passed is False
    assert result.score is False
    assert result.reason is not None
    assert result.metadata["num_questions"] == 0


@pytest.mark.asyncio
async def test_wrong_bullet_style_fails() -> None:
    evaluator = FollowupFormatEvaluator(
        min_questions=3,
        bullet_style="dash",
    )

    prediction = EvalPrediction(
        example_id=1,
        output=(
            "* What environment are you using?\n"
            "* What are you trying to deploy?\n"
            "* Are you using Docker Compose?"
        ),
    )

    result = await evaluator.evaluate(
        EvalExample(id=1, input="Docker"),
        prediction,
    )

    assert result.passed is False
    assert result.reason is not None
