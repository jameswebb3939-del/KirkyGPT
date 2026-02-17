from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register custom pytest command-line options.
    
    Args:
        parser: pytest Parser to register options with.
    """
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
    Get the project root directory path.
    
    Resolves the root by assuming tests/ is at <root>/tests/.
    
    Returns:
        Project root Path.
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path, request: pytest.FixtureRequest) -> Path:
    """
    Get the data directory path.
    
    Can be overridden with --data-dir flag, otherwise defaults to <project_root>/data.
    
    Args:
        project_root: Project root fixture.
        request: pytest request object.
    
    Returns:
        Data directory Path.
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
      1) --sft-jsonl CLI flag
      2) SFT_JSONL environment variable
      3) <data_dir>/sft_followups.jsonl
    
    Args:
        data_dir: Data directory fixture.
        request: pytest request object.
    
    Returns:
        Path to SFT JSONL file.
    
    Raises:
        FileNotFoundError: If file not found.
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
    Iterate over JSONL file, yielding each row as a dict.
    
    Args:
        path: Path to JSONL file.
    
    Returns:
        Iterator over parsed JSON objects.
    
    Raises:
        ValueError: If JSON is malformed or object is not a dict.
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
    
    Args:
        sft_jsonl_path: Path to SFT JSONL file fixture.
    
    Returns:
        List of parsed JSONL rows as dicts.
    
    Raises:
        ValueError: If no rows found in file.
    """
    rows = list(iter_jsonl(sft_jsonl_path))
    if not rows:
        raise ValueError(f"No rows found in JSONL file: {sft_jsonl_path}")
    return rows
