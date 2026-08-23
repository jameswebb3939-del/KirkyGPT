from __future__ import annotations

import pytest

from llm_followups.eval.core.models import (
    EvalExample,
    EvalPrediction,
    EvaluationResult,
)
from llm_followups.eval.core.runner import EvaluationRunner


class FakeTarget:
    async def generate(self, example: EvalExample) -> EvalPrediction:
        return EvalPrediction(
            example_id=example.id,
            output=f"generated:{example.input}",
        )


class FakeEvaluator:
    def __init__(self, name: str, score: float, passed: bool) -> None:
        self._name = name
        self._score = score
        self._passed = passed

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        example: EvalExample,
        prediction: EvalPrediction,
    ) -> EvaluationResult:
        return EvaluationResult(
            evaluator=self.name,
            score=self._score,
            passed=self._passed,
            metadata={
                "input_seen": example.input,
                "output_seen": prediction.output,
            },
        )


@pytest.mark.asyncio
async def test_runner_uses_target_and_evaluator_black_boxes() -> None:
    runner = EvaluationRunner(
        target=FakeTarget(),
        evaluators=[FakeEvaluator("quality", 1.0, True)],
    )

    results = await runner.run(
        [EvalExample(id=1, input="hello")]
    )

    assert len(results) == 1

    item = results[0]
    assert item.example.id == 1
    assert item.prediction.example_id == 1
    assert item.prediction.output == "generated:hello"

    assert len(item.evaluations) == 1
    evaluation = item.evaluations[0]

    assert evaluation.evaluator == "quality"
    assert evaluation.score == 1.0
    assert evaluation.passed is True
    assert evaluation.metadata["input_seen"] == "hello"
    assert evaluation.metadata["output_seen"] == "generated:hello"


@pytest.mark.asyncio
async def test_runner_runs_multiple_evaluators_in_order() -> None:
    runner = EvaluationRunner(
        target=FakeTarget(),
        evaluators=[
            FakeEvaluator("first", 1.0, True),
            FakeEvaluator("second", 0.0, False),
        ],
    )

    results = await runner.run(
        [EvalExample(id=7, input="test")]
    )

    names = [
        evaluation.evaluator
        for evaluation in results[0].evaluations
    ]

    assert names == ["first", "second"]


@pytest.mark.asyncio
async def test_runner_supports_no_evaluators() -> None:
    runner = EvaluationRunner(
        target=FakeTarget(),
        evaluators=[],
    )

    results = await runner.run(
        [EvalExample(id=1, input="hello")]
    )

    assert len(results) == 1
    assert results[0].prediction.output == "generated:hello"
    assert results[0].evaluations == []
