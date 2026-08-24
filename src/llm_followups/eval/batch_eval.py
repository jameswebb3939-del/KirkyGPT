from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_followups.eval.core.runner import EvaluationRunner
from llm_followups.eval.core.summary import summarise_results
from llm_followups.eval.datasets.jsonl import JSONLDatasetSource
from llm_followups.eval.evaluators.format import (
    FollowupFormatEvaluator,
    RawFollowupFormatEvaluator,
)
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
    bullet_style: str = "either",
) -> dict[str, Any]:
    if model_path is not None:
        os.environ["MODEL_PATH"] = str(model_path)

    settings = get_settings()
    source = JSONLDatasetSource(data_path, limit=limit)

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
        RawFollowupFormatEvaluator(
            min_questions=min_questions,
            bullet_style=bullet_style,  # type: ignore[arg-type]
        ),
        FollowupFormatEvaluator(
            min_questions=min_questions,
            bullet_style=bullet_style,  # type: ignore[arg-type]
        ),
        RubricEvaluator(rubric=COHERENCE_RUBRIC, judge=judge),
        RubricEvaluator(rubric=RELEVANCE_RUBRIC, judge=judge),
    ]

    runner = EvaluationRunner(target=target, evaluators=evaluators)
    results = await runner.run(source.load())

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = CSVReporter(output_dir / "results.csv").write(results)
    json_path = JSONReporter(output_dir / "results.json").write(results)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summarise_results(results),
    }
