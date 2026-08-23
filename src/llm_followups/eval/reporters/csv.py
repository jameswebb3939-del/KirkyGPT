from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from llm_followups.eval.core.models import EvaluatedExample


class CSVReporter:
    """Writes generic evaluated examples to CSV."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, results: Sequence[EvaluatedExample]) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "id",
                    "input",
                    "expected_output",
                    "output",
                    "raw_output",
                    "prediction_metadata",
                    "evaluations",
                ],
            )
            writer.writeheader()

            for item in results:
                writer.writerow(
                    {
                        "id": item.example.id,
                        "input": item.example.input,
                        "expected_output": item.example.expected_output or "",
                        "output": item.prediction.output,
                        "raw_output": item.prediction.raw_output or "",
                        "prediction_metadata": json.dumps(
                            item.prediction.metadata,
                            ensure_ascii=False,
                        ),
                        "evaluations": json.dumps(
                            [asdict(result) for result in item.evaluations],
                            ensure_ascii=False,
                        ),
                    }
                )

        return self._path
