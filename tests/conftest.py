from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sft-jsonl",
        action="store",
        default=None,
        help="Path to SFT JSONL file (overrides env SFT_JSONL and default path).",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Path to data directory (default: <project_root>/data).",
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Resolve repo root from tests/ directory.

    Assumes tests/ is at: <root>/tests/
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path, request: pytest.FixtureRequest) -> Path:
    """
    Default data dir: <project_root>/data
    Override with: --data-dir
    """
    override = request.config.getoption("--data-dir")
    if override:
        return Path(override).expanduser().resolve()
    return (project_root / "data").resolve()


@pytest.fixture(scope="session")
def sft_jsonl_path(data_dir: Path, request: pytest.FixtureRequest) -> Path:
    """
    Resolve the SFT JSONL file path.

    Priority:
      1) --sft-jsonl
      2) env var SFT_JSONL
      3) <data_dir>/sft.jsonl
    """
    cli_path = request.config.getoption("--sft-jsonl")
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
    else:
        env_path = os.environ.get("SFT_JSONL")
        if env_path:
            path = Path(env_path).expanduser().resolve()
        else:
            path = (data_dir / "sft_followups.jsonl").resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"SFT JSONL file not found: {path}\n"
            f"Set it via --sft-jsonl <path> or env SFT_JSONL."
        )
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Iterate JSONL rows as dicts.
    Skips empty lines.
    Raises ValueError with line number on JSON parse issues.
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
                raise ValueError(f"Expected JSON object (dict) on line {lineno}, got {type(obj).__name__}")
            yield obj


@pytest.fixture(scope="session")
def sft_rows(sft_jsonl_path: Path) -> list[dict[str, Any]]:
    """
    Load all JSONL rows into memory.
    """
    rows = list(iter_jsonl(sft_jsonl_path))
    if not rows:
        raise ValueError(f"No rows found in JSONL file: {sft_jsonl_path}")
    return rows
