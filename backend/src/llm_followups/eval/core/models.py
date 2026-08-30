from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    """One evaluation input plus optional reference data and metadata."""

    id: int
    input: str
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalPrediction:
    """Output produced by the system under evaluation."""

    example_id: int
    output: str
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Provider-neutral result returned by any evaluator."""

    evaluator: str
    score: float | bool | None
    passed: bool | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatedExample:
    """An input, its generated prediction, and all evaluation results."""

    example: EvalExample
    prediction: EvalPrediction
    evaluations: list[EvaluationResult]


@dataclass(frozen=True)
class JudgeResult:
    """Normalized output from an LLM-as-a-judge provider."""

    score: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
