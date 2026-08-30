from __future__ import annotations

from collections.abc import Sequence

from llm_followups.eval.core.models import EvalExample, EvaluatedExample
from llm_followups.eval.core.protocols import Evaluator, Target


class EvaluationRunner:
    """Orchestrates a target and a set of evaluators without knowing implementations."""

    def __init__(self, *, target: Target, evaluators: Sequence[Evaluator]) -> None:
        self._target = target
        self._evaluators = list(evaluators)

    async def run_example(self, example: EvalExample) -> EvaluatedExample:
        prediction = await self._target.generate(example)

        evaluations = []
        for evaluator in self._evaluators:
            result = await evaluator.evaluate(example, prediction)
            evaluations.append(result)

        return EvaluatedExample(
            example=example,
            prediction=prediction,
            evaluations=evaluations,
        )

    async def run(self, examples: Sequence[EvalExample]) -> list[EvaluatedExample]:
        results: list[EvaluatedExample] = []

        # Keep this sequential first. Concurrency can be added later without changing
        # Target/Evaluator contracts.
        for example in examples:
            results.append(await self.run_example(example))

        return results
