from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Read JSON objects from a JSONL file.

    Blank lines are ignored.

    Raises:
        FileNotFoundError:
            If the dataset does not exist.

        ValueError:
            If the file contains invalid JSON,
            non-object rows, or no examples.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw in enumerate(
            file,
            start=1,
        ):
            line = raw.strip()

            if not line:
                continue

            try:
                value = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    "Expected JSON object on line "
                    f"{line_number}, got "
                    f"{type(value).__name__}"
                )

            rows.append(value)

    if not rows:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Deterministically split rows into
    training and evaluation sets.

    The source list is not modified.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(
            "eval_fraction must be "
            "between 0 and 1"
        )

    if len(rows) < 2:
        raise ValueError(
            "At least 2 rows are required "
            "for train/eval splitting"
        )

    shuffled = list(rows)

    random.Random(seed).shuffle(
        shuffled
    )

    eval_count = max(
        1,
        round(
            len(shuffled)
            * eval_fraction
        ),
    )

    if eval_count >= len(shuffled):
        raise ValueError(
            "Evaluation split leaves "
            "no training rows"
        )

    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]

    return train_rows, eval_rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Write rows to JSONL.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            file.write("\n")