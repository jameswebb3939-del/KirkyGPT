from __future__ import annotations

from typing import Literal

from llm_followups.eval.core.models import EvalExample, EvalPrediction, EvaluationResult
from llm_followups.tuning.validate import validate_followup_list


class _BaseFollowupFormatEvaluator:
    def __init__(
        self,
        *,
        min_questions: int = 3,
        bullet_style: Literal["dash", "asterisk", "either"] = "either",
    ) -> None:
        self._min_questions = min_questions
        self._bullet_style = bullet_style

    def _text(self, prediction: EvalPrediction) -> str | None:
        raise NotImplementedError

    async def evaluate(
        self,
        example: EvalExample,
        prediction: EvalPrediction,
    ) -> EvaluationResult:
        del example
        text = self._text(prediction)
        if text is None:
            return EvaluationResult(
                evaluator=self.name,
                score=False,
                passed=False,
                reason="Prediction did not contain text for this evaluator.",
                metadata={"num_questions": 0, "normalized_text": None},
            )

        result = validate_followup_list(
            text=text,
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


class FollowupFormatEvaluator(_BaseFollowupFormatEvaluator):
    """Evaluate the final, post-guardrail output."""

    @property
    def name(self) -> str:
        return "followup_format"

    def _text(self, prediction: EvalPrediction) -> str | None:
        return prediction.output


class RawFollowupFormatEvaluator(_BaseFollowupFormatEvaluator):
    """Evaluate raw model output before repair, trimming, or fallback."""

    @property
    def name(self) -> str:
        return "raw_followup_format"

    def _text(self, prediction: EvalPrediction) -> str | None:
        return prediction.raw_output
