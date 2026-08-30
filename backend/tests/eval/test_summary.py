from __future__ import annotations

import pytest

from llm_followups.eval.core.models import (
    EvalExample,
    EvalPrediction,
    EvaluatedExample,
    EvaluationResult,
)
from llm_followups.eval.core.summary import summarise_results


def make_results() -> list[EvaluatedExample]:
    return [
        EvaluatedExample(
            example=EvalExample(
                id=1,
                input="Example A",
            ),
            prediction=EvalPrediction(
                example_id=1,
                output="output A",
                metadata={
                    "latency_ms": 100,
                    "used_fallback": False,
                    "used_repair": True,
                },
            ),
            evaluations=[
                EvaluationResult(
                    evaluator="followup_format",
                    score=True,
                    passed=True,
                ),
                EvaluationResult(
                    evaluator="coherence",
                    score=3.0,
                    passed=True,
                ),
            ],
        ),
        EvaluatedExample(
            example=EvalExample(
                id=2,
                input="Example B",
            ),
            prediction=EvalPrediction(
                example_id=2,
                output="output B",
                metadata={
                    "latency_ms": 300,
                    "used_fallback": True,
                    "used_repair": False,
                },
            ),
            evaluations=[
                EvaluationResult(
                    evaluator="followup_format",
                    score=False,
                    passed=False,
                ),
                EvaluationResult(
                    evaluator="coherence",
                    score=1.0,
                    passed=False,
                ),
            ],
        ),
    ]


def test_summarise_results_calculates_expected_metrics() -> None:
    summary = summarise_results(make_results())

    assert summary["count_example"] == 2
    assert summary["average_latency_ms"] == pytest.approx(200.0)
    assert summary["fallback_rate"] == pytest.approx(50.0)
    assert summary["repair_rate"] == pytest.approx(50.0)
    assert summary["format_valid_percentage"] == pytest.approx(50.0)

    format_summary = summary["evaluators"]["followup_format"]
    assert format_summary["score_count"] == 0
    assert format_summary["average_score"] is None
    assert format_summary["pass_count"] == 1
    assert format_summary["pass_rate_percentage"] == pytest.approx(50.0)

    coherence_summary = summary["evaluators"]["coherence"]
    assert coherence_summary["score_count"] == 2
    assert coherence_summary["average_score"] == pytest.approx(2.0)
    assert coherence_summary["pass_count"] == 1
    assert coherence_summary["pass_rate_percentage"] == pytest.approx(50.0)


def test_summarise_results_handles_empty_input() -> None:
    summary = summarise_results([])

    assert summary == {
        "count_example": 0,
        "average_latency_ms": 0.0,
        "fallback_rate": 0.0,
        "repair_rate": 0.0,
        "evaluators": {},
    }


def test_boolean_scores_are_not_treated_as_numeric_scores() -> None:
    result = EvaluatedExample(
        example=EvalExample(id=1, input="test"),
        prediction=EvalPrediction(
            example_id=1,
            output="output",
        ),
        evaluations=[
            EvaluationResult(
                evaluator="boolean_check",
                score=True,
                passed=True,
            )
        ],
    )

    summary = summarise_results([result])
    evaluator_summary = summary["evaluators"]["boolean_check"]

    assert evaluator_summary["score_count"] == 0
    assert evaluator_summary["average_score"] is None
    assert evaluator_summary["pass_rate_percentage"] == pytest.approx(100.0)
