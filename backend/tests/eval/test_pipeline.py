from __future__ import annotations

import json

import pytest

from llm_followups.eval.core.models import EvalExample, EvalPrediction, JudgeResult
from llm_followups.eval.core.runner import EvaluationRunner
from llm_followups.eval.core.summary import summarise_results
from llm_followups.eval.datasets.jsonl import JSONLDatasetSource
from llm_followups.eval.evaluators.format import FollowupFormatEvaluator
from llm_followups.eval.evaluators.rubric import Rubric, RubricEvaluator
from llm_followups.eval.reporters.json import JSONReporter


class FakeTarget:
    async def generate(
        self,
        example: EvalExample,
    ) -> EvalPrediction:
        return EvalPrediction(
            example_id=example.id,
            output=(
                "- Are you mourning Charlie for the Kirkiversary?\n"
                "- Do you want the Erika timeline or the beneficiary list?\n"
                "- Should we open with the roof-shot or the cover-up?"
            ),
            metadata={
                "latency_ms": 10,
                "used_fallback": False,
                "used_repair": False,
            },
        )


class FakeJudge:
    async def judge(
        self,
        *,
        instructions: str,
        input_text: str,
        output_text: str,
    ) -> JudgeResult:
        assert instructions
        assert input_text == "Help me with Kirk"
        assert output_text.startswith("- Are you mourning")

        return JudgeResult(
            score=3.0,
            reason="Relevant",
            metadata={"judge_provider": "fake"},
        )


@pytest.mark.asyncio
async def test_black_box_pipeline_end_to_end_without_external_services(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": 1,
                "prompt": "Help me with Kirk",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source = JSONLDatasetSource(dataset_path)

    rubric = Rubric(
        name="answer_relevance",
        description="Evaluate answer relevance.",
        score_levels={
            0: "irrelevant",
            3: "directly relevant",
        },
        pass_threshold=2.0,
    )

    runner = EvaluationRunner(
        target=FakeTarget(),
        evaluators=[
            FollowupFormatEvaluator(
                min_questions=3,
                bullet_style="dash",
            ),
            RubricEvaluator(
                rubric=rubric,
                judge=FakeJudge(),
            ),
        ],
    )

    examples = source.load()
    results = await runner.run(examples)

    assert len(results) == 1
    assert [
        evaluation.evaluator
        for evaluation in results[0].evaluations
    ] == [
        "followup_format",
        "answer_relevance",
    ]

    assert results[0].evaluations[0].passed is True
    assert results[0].evaluations[1].score == 3.0
    assert results[0].evaluations[1].passed is True

    summary = summarise_results(results)

    assert summary["count_example"] == 1
    assert summary["format_valid_percentage"] == pytest.approx(100.0)
    assert summary["evaluators"]["answer_relevance"]["average_score"] == pytest.approx(3.0)
    assert summary["evaluators"]["answer_relevance"]["pass_rate_percentage"] == pytest.approx(100.0)

    output_path = tmp_path / "results.json"
    JSONReporter(output_path).write(results)

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved[0]["example"]["input"] == "Help me with Kirk"
    assert saved[0]["evaluations"][0]["evaluator"] == "followup_format"
    assert saved[0]["evaluations"][1]["evaluator"] == "answer_relevance"