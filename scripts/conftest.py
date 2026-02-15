from __future__ import annotations
from pathlib import Path
import json
import pytest
from typing import Any, Iterator

def project_root() -> Path:
    path = Path(__file__).resolve().parents[2]
    return path

def data_dir(project_root: Path) -> Path:
    if project_root is None:
        raise ValueError("Project root is none.")
    return project_root / "data"

def sft_jsonl_path(data_dir: Path) -> Path:
    if data_dir is None:
        raise ValueError("Data directory is none.")
    path = data_dir / "sft_followups.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"SFT JSONL file not found: {path}")
    return path

def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path is None:
        raise ValueError("Path is None.")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict, got {type(obj).__name__}")
            yield obj

def sft_rows(sft_jsonl_path: Path) -> list[dict[str, Any]]:
    parsed_rows = []
    for obj in iter_jsonl(sft_jsonl_path):
        parsed_rows.append(obj)
    return parsed_rows

def validate_sft_row(row: dict[str, Any]) -> None:
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
        if "role" not in msg:
            raise ValueError("Message missing required field 'role'")
        if "content" not in msg:
            raise ValueError("Message missing required field 'content'")
        if msg["role"] not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {msg['role']}")
        if not msg["content"] or not isinstance(msg["content"], str):
            raise ValueError(f"Content must be non-empty string, got {type(msg['content']).__name__}")