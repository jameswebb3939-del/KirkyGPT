from __future__ import annotations

from typing import Literal

from llm_followups.eval.core.models import EvalExample, EvalPrediction, EvaluationResult
from llm_followups.tuning.validate import validate_followup_list


class FollowupFormatEvaluator:
    """Adapter around the project's existing follow-up format validator."""

    def __init__(self, *, min_questions: int = 3, bullet_style: Literal["dash", "asterisk", "either"] = "either") -> None:
        self._min_questions = min_questions
        self._bullet_style = bullet_style

    @property
    def name(self) -> str:
        return "followup_format"

    async def evaluate(self, example: EvalExample, prediction: EvalPrediction) -> EvaluationResult:
        del example  # This evaluator only needs the prediction.

        result = validate_followup_list(
            text=prediction.output,
            min_questions=self._min_questions,
            bullet_style=self._bullet_style,
            require_question_mark=True,
            forbid_extra_text=True,
        )

        return EvaluationResult(
            evaluator=self.name,
            score=result.ok,
            passed=result.ok,
            reason="; ".join(result.errors) if result.errors else None,
            metadata={
                "num_questions": result.num_items,
                "normalized_text": result.normalized_text,
            },
        )
