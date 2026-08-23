from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from llm_followups.eval.core.runner import EvaluationRunner
from llm_followups.eval.core.summary import summarise_results
from llm_followups.eval.datasets.jsonl import JSONLDatasetSource
from llm_followups.eval.evaluators.format import FollowupFormatEvaluator
from llm_followups.eval.evaluators.rubric import RubricEvaluator
from llm_followups.eval.judges.openai import OpenAIJudge
from llm_followups.eval.reporters.csv import CSVReporter
from llm_followups.eval.reporters.json import JSONReporter
from llm_followups.eval.rubrics import COHERENCE_RUBRIC, RELEVANCE_RUBRIC
from llm_followups.eval.targets.llm_runtime import LLMRuntimeTarget
from llm_followups.server.llm_runtime import LLMRuntime
from llm_followups.utils.config import get_settings


async def run_batch_evaluation(
    data_path: Path,
    output_dir: Path,
    model_path: Path | None = None,
    limit: int | None = None,
    min_questions: int = 3,
    bullet_style: Literal["dash", "asterisk", "either"] = "either",
) -> dict[str, Any]:
    """
    Composition root for the evaluation pipeline.

    Every stage behind this function is accessed through a small contract:
    dataset source -> target -> evaluators/judge -> reporters.
    """

    if model_path is not None:
        # Existing LLMRuntime supports a local model override through MODEL_PATH.
        os.environ["MODEL_PATH"] = str(model_path)

    settings = get_settings()

    source = JSONLDatasetSource(data_path, limit=limit)
    examples = source.load()

    runtime = LLMRuntime(settings=settings)
    await runtime.load()

    target = LLMRuntimeTarget(
        runtime,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )

    judge = OpenAIJudge(
        client=OpenAI(),
        model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
    )

    evaluators = [
        FollowupFormatEvaluator(
            min_questions=min_questions,
            bullet_style=bullet_style,
        ),
        RubricEvaluator(rubric=COHERENCE_RUBRIC, judge=judge),
        RubricEvaluator(rubric=RELEVANCE_RUBRIC, judge=judge),
    ]

    runner = EvaluationRunner(
        target=target,
        evaluators=evaluators,
    )

    results = await runner.run(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    reporters = [
        CSVReporter(csv_path),
        JSONReporter(json_path),
    ]

    for reporter in reporters:
        reporter.write(results)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summarise_results(results),
    }
