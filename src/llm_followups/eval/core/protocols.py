from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from llm_followups.eval.core.models import EvalExample, EvalPrediction, EvaluatedExample, EvaluationResult, JudgeResult


class DatasetSource(Protocol):
    """Black-box source of evaluation examples."""

    def load(self) -> Sequence[EvalExample]:
        ...


class Target(Protocol):
    """Black-box system under evaluation."""

    async def generate(self, example: EvalExample) -> EvalPrediction:
        ...


class Evaluator(Protocol):
    """Black-box evaluator for a target prediction."""

    @property
    def name(self) -> str:
        ...

    async def evaluate(self, example: EvalExample, prediction: EvalPrediction) -> EvaluationResult:
        ...


class Judge(Protocol):
    """Black-box LLM judge used by rubric-based evaluators."""

    async def judge(self, *, instructions: str, input_text: str, output_text: str) -> JudgeResult:
        ...


class Reporter(Protocol):
    """Black-box result sink."""

    def write(self, results: Sequence[EvaluatedExample]) -> Path | None:
        ...
