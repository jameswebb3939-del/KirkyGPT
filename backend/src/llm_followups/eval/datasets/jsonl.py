from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_followups.eval.core.models import EvalExample


class JSONLDatasetSource:
    """Loads evaluation examples from the JSONL shapes already used by the project."""

    def __init__(self, path: Path, *, limit: int | None = None) -> None:
        self._path = path
        self._limit = limit

    def load(self) -> list[EvalExample]:
        examples: list[EvalExample] = []

        with self._path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Line {line_number}: expected JSON object, got {type(row).__name__}"
                    )

                example = self._parse_row(row=row, fallback_id=len(examples))
                if example is None:
                    continue

                examples.append(example)

                if self._limit is not None and len(examples) >= self._limit:
                    break

        return examples

    def _parse_row(self, *, row: dict[str, Any], fallback_id: int) -> EvalExample | None:
        example_id = self._coerce_id(row.get("id"), fallback_id=fallback_id)

        input_text: str | None = None
        expected_output: str | None = None

        messages = row.get("messages")
        if isinstance(messages, list):
            user_messages = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            assistant_messages = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]

            if user_messages:
                value = user_messages[-1].get("content")
                if isinstance(value, str):
                    input_text = value

            if assistant_messages:
                value = assistant_messages[-1].get("content")
                if isinstance(value, str):
                    expected_output = value

        if input_text is None:
            value = row.get("user_message")
            if isinstance(value, str):
                input_text = value

        if input_text is None:
            value = row.get("prompt")
            if isinstance(value, str):
                input_text = value

        if input_text is None or not input_text.strip():
            return None

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "messages", "user_message", "prompt"}
        }

        return EvalExample(
            id=example_id,
            input=input_text.strip(),
            expected_output=expected_output.strip() if expected_output else None,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_id(value: Any, *, fallback_id: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass

        return fallback_id
