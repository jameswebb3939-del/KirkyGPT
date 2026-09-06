from __future__ import annotations

import csv
import json

from llm_followups.eval.core.models import (
    EvalExample,
    EvalPrediction,
    EvaluatedExample,
    EvaluationResult,
)
from llm_followups.eval.reporters.csv import CSVReporter
from llm_followups.eval.reporters.json import JSONReporter


def make_results() -> list[EvaluatedExample]:
    return [
        EvaluatedExample(
            example=EvalExample(
                id=1,
                input="Help me with Kirk",
                expected_output="- Are you mourning or hunting?",
                metadata={"source": "test"},
            ),
            prediction=EvalPrediction(
                example_id=1,
                output="- Are you mourning Charlie for the Kirkiversary?",
                raw_output="raw output",
                metadata={
                    "latency_ms": 123,
                    "used_repair": False,
                },
            ),
            evaluations=[
                EvaluationResult(
                    evaluator="coherence",
                    score=3.0,
                    passed=True,
                    reason="Clear",
                    metadata={"judge_provider": "fake"},
                )
            ],
        )
    ]


def test_json_reporter_writes_complete_result_model(tmp_path) -> None:
    path = tmp_path / "results.json"

    returned_path = JSONReporter(path).write(make_results())

    assert returned_path == path
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))

    assert len(data) == 1
    assert data[0]["example"]["id"] == 1
    assert data[0]["example"]["input"] == "Help me with Kirk"
    assert data[0]["prediction"]["output"] == "- Are you mourning Charlie for the Kirkiversary?"
    assert data[0]["prediction"]["metadata"]["latency_ms"] == 123
    assert data[0]["evaluations"][0]["evaluator"] == "coherence"
    assert data[0]["evaluations"][0]["score"] == 3.0


def test_csv_reporter_writes_serialized_metadata_and_evaluations(
    tmp_path,
) -> None:
    path = tmp_path / "results.csv"

    returned_path = CSVReporter(path).write(make_results())

    assert returned_path == path
    assert path.exists()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    row = rows[0]
    assert row["id"] == "1"
    assert row["input"] == "Help me with Kirk"
    assert row["expected_output"] == "- Are you mourning or hunting?"
    assert row["output"] == "- Are you mourning Charlie for the Kirkiversary?"
    assert row["raw_output"] == "raw output"

    prediction_metadata = json.loads(row["prediction_metadata"])
    assert prediction_metadata["latency_ms"] == 123
    assert prediction_metadata["used_repair"] is False

    evaluations = json.loads(row["evaluations"])
    assert len(evaluations) == 1
    assert evaluations[0]["evaluator"] == "coherence"
    assert evaluations[0]["score"] == 3.0
    assert evaluations[0]["passed"] is True