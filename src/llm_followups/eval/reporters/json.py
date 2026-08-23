from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from llm_followups.eval.core.models import EvaluatedExample


class JSONReporter:
    """Writes the complete provider-neutral result model to JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, results: Sequence[EvaluatedExample]) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("w", encoding="utf-8") as file:
            json.dump(
                [asdict(item) for item in results],
                file,
                indent=2,
                ensure_ascii=False,
            )

        return self._path
