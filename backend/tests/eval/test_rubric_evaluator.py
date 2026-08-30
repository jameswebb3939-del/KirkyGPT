from __future__ import annotations

import pytest

from llm_followups.eval.core.models import (
    EvalExample,
    EvalPrediction,
    JudgeResult,
)
from llm_followups.eval.evaluators.rubric import (
    Rubric,
    RubricEvaluator,
)


class FakeJudge:
    def __init__(
        self,
        *,
        score: float,
        reason: str = "judge reason",
    ) -> None:
        self._score = score
        self._reason = reason
        self.calls: list[dict[str, str]] = []

    async def judge(
        self,
        *,
        instructions: str,
        input_text: str,
        output_text: str,
    ) -> JudgeResult:
        self.calls.append(
            {
                "instructions": instructions,
                "input_text": input_text,
                "output_text": output_text,
            }
        )

        return JudgeResult(
            score=self._score,
            reason=self._reason,
            metadata={"judge_provider": "fake"},
        )


def make_rubric(
    *,
    pass_threshold: float | None = 2.0,
) -> Rubric:
    return Rubric(
        name="coherence",
        description="Evaluate coherence",
        score_levels={
            0: "bad",
            1: "weak",
            2: "good",
            3: "excellent",
        },
        pass_threshold=pass_threshold,
    )


@pytest.mark.asyncio
async def test_rubric_evaluator_uses_judge() -> None:
    judge = FakeJudge(score=3.0, reason="Very coherent")

    evaluator = RubricEvaluator(
        rubric=make_rubric(),
        judge=judge,
    )

    result = await evaluator.evaluate(
        EvalExample(id=1, input="Explain Docker"),
        EvalPrediction(example_id=1, output="A clear answer"),
    )

    assert result.evaluator == "coherence"
    assert result.score == 3.0
    assert result.passed is True
    assert result.reason == "Very coherent"
    assert result.metadata["judge_provider"] == "fake"

    assert len(judge.calls) == 1
    assert judge.calls[0]["input_text"] == "Explain Docker"
    assert judge.calls[0]["output_text"] == "A clear answer"
    assert "Evaluate coherence" in judge.calls[0]["instructions"]
    assert "3 = excellent" in judge.calls[0]["instructions"]


@pytest.mark.asyncio
async def test_rubric_evaluator_fails_below_threshold() -> None:
    evaluator = RubricEvaluator(
        rubric=make_rubric(pass_threshold=2.0),
        judge=FakeJudge(score=1.0),
    )

    result = await evaluator.evaluate(
        EvalExample(id=1, input="Explain Docker"),
        EvalPrediction(example_id=1, output="Poor answer"),
    )

    assert result.score == 1.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_rubric_without_threshold_has_no_pass_fail() -> None:
    evaluator = RubricEvaluator(
        rubric=make_rubric(pass_threshold=None),
        judge=FakeJudge(score=2.0),
    )

    result = await evaluator.evaluate(
        EvalExample(id=1, input="Explain Docker"),
        EvalPrediction(example_id=1, output="Answer"),
    )

    assert result.score == 2.0
    assert result.passed is None
