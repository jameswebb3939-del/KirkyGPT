from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from llm_followups.eval.core.models import EvaluatedExample


def summarise_results(results: Sequence[EvaluatedExample]) -> dict[str, Any]:
    """Build a provider-neutral batch summary."""

    count = len(results)
    latencies: list[float] = []
    fallback_count = 0
    repair_count = 0

    scores_by_evaluator: dict[str, list[float]] = defaultdict(list)
    pass_values_by_evaluator: dict[str, list[bool]] = defaultdict(list)

    for item in results:
        metadata = item.prediction.metadata

        latency = metadata.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))

        if metadata.get("used_fallback") is True:
            fallback_count += 1

        if metadata.get("used_repair") is True:
            repair_count += 1

        for evaluation in item.evaluations:
            score = evaluation.score
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                scores_by_evaluator[evaluation.evaluator].append(float(score))

            if evaluation.passed is not None:
                pass_values_by_evaluator[evaluation.evaluator].append(evaluation.passed)

    evaluator_summary: dict[str, dict[str, float | int | None]] = {}
    names = set(scores_by_evaluator) | set(pass_values_by_evaluator)

    for name in sorted(names):
        scores = scores_by_evaluator.get(name, [])
        pass_values = pass_values_by_evaluator.get(name, [])
        evaluator_summary[name] = {
            "score_count": len(scores),
            "average_score": (sum(scores) / len(scores)) if scores else None,
            "pass_count": sum(pass_values) if pass_values else 0,
            "pass_rate_percentage": (
                sum(pass_values) / len(pass_values) * 100 if pass_values else None
            ),
        }

    summary: dict[str, Any] = {
        "count_example": count,
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "fallback_rate": (fallback_count / count * 100) if count else 0.0,
        "repair_rate": (repair_count / count * 100) if count else 0.0,
        "evaluators": evaluator_summary,
    }

    # Compatibility/convenience field for the current project.
    format_summary = evaluator_summary.get("followup_format")
    if format_summary is not None:
        summary["format_valid_percentage"] = format_summary["pass_rate_percentage"] or 0.0

    return summary
