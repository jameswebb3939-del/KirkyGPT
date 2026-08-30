from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest


def project_root() -> Path:
    """
    Return the root directory of the project.
    """
    # .../llm_followups/tests/conftest.py -> parents[1] = repo root
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """
    Pytest fixture returning the path to the data directory.
    """
    return project_root() / "data"


@pytest.fixture(scope="session")
def sft_jsonl_path(data_dir: Path) -> Path:
    """
    Pytest fixture returning the path to the SFT JSONL file.
    Raises FileNotFoundError if the file does not exist.
    """
    path = data_dir / "sft_followups.jsonl"
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"SFT JSONL file not found: {path}")
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over a JSONL file, yielding each row as a dict.
    Raises ValueError for invalid JSON or non-object rows.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno}: expected JSON object, got {type(obj).__name__}")
            yield obj


@pytest.fixture(scope="session")
def sft_rows(sft_jsonl_path: Path) -> list[dict[str, Any]]:
    """
    Pytest fixture returning all rows from the SFT JSONL file as a list of dicts.
    Raises ValueError if no rows are found.
    """
    rows = list(iter_jsonl(sft_jsonl_path))
    if not rows:
        raise ValueError(f"No rows found in {sft_jsonl_path}")
    return rows