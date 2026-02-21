from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from src.llm_followups.server.schemas import ChatRequest
from src.llm_followups.tuning.validate import validate_followup_list


def get_last_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    """
    Return the last non-empty assistant message text from a list of chat messages.
    """
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def test_jsonl_rows_are_valid_json(sft_rows: list[dict[str, Any]]) -> None:
    """
    Test that all rows in the SFT JSONL file are valid JSON objects (dicts).
    """
    assert isinstance(sft_rows, list)
    assert len(sft_rows) > 0
    for row in sft_rows:
        assert isinstance(row, dict)


def test_each_row_matches_chatrequest_schema(sft_rows: list[dict[str, Any]]) -> None:
    """
    Test that each row matches the ChatRequest schema using pydantic validation.
    Raises AssertionError if schema validation fails.
    """
    for row in sft_rows:
        try:
            ChatRequest(**row)
        except ValidationError as e:
            raise AssertionError(f"Schema validation failed: {e}") from e


def test_assistant_messages_are_followup_bullets(sft_rows: list[dict[str, Any]]) -> None:
    """
    Test that the assistant's last message in each row is a valid followup bullet list.
    Requires at least 3 questions, question marks, and forbids extra text.
    """
    for row in sft_rows:
        messages = row["messages"]
        assistant_text = get_last_assistant_text(messages)
        assert assistant_text is not None

        result = validate_followup_list(
            text=assistant_text,
            min_questions=3,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True,
        )
        assert result.ok, result.errors


def test_no_numbered_lists_in_assistant_output(sft_rows: list[dict[str, Any]]) -> None:
    """
    Test that the assistant's output does not contain numbered lists (only bullets allowed).
    """
    numbered = re.compile(r"^\s*\d+[\.\)]\s+")
    for row in sft_rows:
        assistant_text = get_last_assistant_text(row["messages"])
        assert assistant_text is not None
        for line in assistant_text.splitlines():
            assert not numbered.match(line), f"Numbered list line not allowed: {line!r}"


def test_min_questions_respected(sft_rows: list[dict[str, Any]]) -> None:
    """
    Test that the assistant's output contains at least the minimum required number of questions.
    """
    min_q = 3
    for row in sft_rows:
        assistant_text = get_last_assistant_text(row["messages"])
        assert assistant_text is not None

        result = validate_followup_list(
            text=assistant_text,
            min_questions=min_q,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True,
        )
        assert result.ok, result.errors
        assert result.num_items >= min_q