from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Return the repository root (folder containing pyproject.toml).
    This file lives in: <root>/tests/conftest.py
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """
    Return <root>/data
    """
    path = project_root / "data"
    if not path.exists():
        raise FileNotFoundError(f"Data dir not found: {path}")
    return path


@pytest.fixture(scope="session")
def sft_jsonl_path(data_dir: Path) -> Path:
    """
    Return <root>/data/sft_followups.jsonl
    """
    path = data_dir / "sft_followups.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"SFT JSONL file not found: {path}")
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Yield parsed dict rows from a JSONL file. Skips blank lines.
    Raises if a line is not valid JSON or not a JSON object.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
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
    Load all rows once per test session.
    """
    return list(iter_jsonl(sft_jsonl_path))


def validate_sft_row(row: dict[str, Any]) -> None:
    """
    Basic structural validation of a row (helper; tests can call this).
    """
    if not isinstance(row, dict):
        raise TypeError(f"Row must be a dict, got {type(row).__name__}")

    if "messages" not in row:
        raise ValueError("Row missing required field 'messages'")

    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"'messages' must be a list, got {type(messages).__name__}")

    if not messages:
        raise ValueError("'messages' list is empty")

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"Message must be a dict, got {type(msg).__name__}")

        role = msg.get("role")
        content = msg.get("content")

        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be a non-empty string")
