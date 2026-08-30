from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from llm_followups.eval.core.models import EvalExample, EvalPrediction, EvaluationResult

from llm_followups.eval.core.protocols import Judge


@dataclass(frozen=True)
class Rubric:
    """Configuration for a single rubric-scored evaluation dimension."""

    name: str
    description: str
    score_levels: Mapping[int, str] = field(default_factory=dict)
    pass_threshold: float | None = None

    def render_instructions(self) -> str:
        parts = [self.description.strip()]

        if self.score_levels:
            parts.append("Score guide:")
            for score, meaning in sorted(self.score_levels.items()):
                parts.append(f"{score} = {meaning}")

        return "\n".join(parts)


class RubricEvaluator:
    """Generic rubric evaluator that delegates judging to a Judge black box."""

    def __init__(self, *, rubric: Rubric, judge: Judge) -> None:
        self._rubric = rubric
        self._judge = judge

    @property
    def name(self) -> str:
        return self._rubric.name

    async def evaluate(self, example: EvalExample, prediction: EvalPrediction) -> EvaluationResult:
        judge_result = await self._judge.judge(
            instructions=self._rubric.render_instructions(),
            input_text=example.input,
            output_text=prediction.output,
        )

        passed = None
        if self._rubric.pass_threshold is not None:
            passed = judge_result.score >= self._rubric.pass_threshold

        return EvaluationResult(
            evaluator=self.name,
            score=judge_result.score,
            passed=passed,
            reason=judge_result.reason,
            metadata=dict(judge_result.metadata),
        )
