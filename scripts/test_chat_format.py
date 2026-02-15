from __future__ import annotations
from typing import Any
import pytest
import re

from src.llm_followups.server.schemas import ChatRequest
from src.llm_followups.tuning.validate import validate_followup_list

pytestmark = pytest.mark.unit

def assert_row_schema(row: dict[str, Any]) -> None:
    assert isinstance(row, dict)
    assert "messages" in row
    assert isinstance(row["messages"], list)
    assert len(row["messages"]) >= 1
    for m in row["messages"]:
        assert isinstance(m, dict)
        assert "role" in m and "content" in m
        assert isinstance(m["role"], str)
        assert m["role"] in {"user", "assistant"}  # Add "system" if needed
        assert isinstance(m["content"], str)
        assert m["content"].strip() != ""

def get_last_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip() != "":
                return content.strip()
    return None

def test_jsonl_rows_are_valid_json(sft_rows: list[dict[str, Any]]) -> None:
    assert isinstance(sft_rows, list)
    assert len(sft_rows) > 0
    for row in sft_rows:
        assert isinstance(row, dict)
        assert "messages" in row

def test_each_row_matches_chatrequest_schema(sft_rows: list[dict[str, Any]]) -> None:
    for row in sft_rows:
        assert_row_schema(row)
        parsed = ChatRequest(**row)
        assert len(parsed.messages) >= 1

def test_assistant_messages_are_followup_bullets(sft_rows: list[dict[str, Any]]) -> None:
    for row in sft_rows:
        messages = row["messages"]
        assistant_text = get_last_assistant_text(messages)
        assert assistant_text is not None
        result = validate_followup_list(
            text=assistant_text,
            min_questions=3,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True
        )
        assert result.ok, result.errors

def test_no_numbered_lists_in_assistant_output(sft_rows: list[dict[str, Any]]) -> None:
    pattern = re.compile(r"^\s*\d+[\.\)]\s+")
    for idx, row in enumerate(sft_rows):
        assistant_text = get_last_assistant_text(row["messages"])
        if assistant_text is None:
            continue
        for line in assistant_text.splitlines():
            if pattern.match(line):
                pytest.fail(f"Row {idx}: Numbered list found in line: '{line}'")

def test_min_questions_respected(sft_rows: list[dict[str, Any]]) -> None:
    min_q = 3
    for row in sft_rows:
        assistant_text = get_last_assistant_text(row["messages"])
        assert assistant_text is not None
        result = validate_followup_list(
            text=assistant_text,
            min_questions=min_q,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True
        )
        assert result.num_items >= min_q
        assert result.ok, result.errors