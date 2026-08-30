from __future__ import annotations

import json

import pytest

from llm_followups.eval.datasets.jsonl import JSONLDatasetSource


def write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loads_sft_messages(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"

    write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Explain Docker"},
                    {
                        "role": "assistant",
                        "content": "- What do you want to know?",
                    },
                ],
                "source": "sft",
            }
        ],
    )

    source = JSONLDatasetSource(path)
    examples = source.load()

    assert len(examples) == 1
    assert examples[0].id == 0
    assert examples[0].input == "Explain Docker"
    assert examples[0].expected_output == "- What do you want to know?"
    assert examples[0].metadata["source"] == "sft"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", "Explain Redis"),
        ("user_message", "Explain FastAPI"),
    ],
)
def test_loads_supported_plain_input_shapes(
    tmp_path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "eval.jsonl"

    write_jsonl(
        path,
        [
            {
                "id": "12",
                field: value,
                "source": "manual",
            }
        ],
    )

    examples = JSONLDatasetSource(path).load()

    assert len(examples) == 1
    assert examples[0].id == 12
    assert examples[0].input == value
    assert examples[0].expected_output is None
    assert examples[0].metadata["source"] == "manual"


def test_limit_is_respected(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"

    write_jsonl(
        path,
        [
            {"prompt": "one"},
            {"prompt": "two"},
            {"prompt": "three"},
        ],
    )

    examples = JSONLDatasetSource(
        path,
        limit=2,
    ).load()

    assert [example.input for example in examples] == [
        "one",
        "two",
    ]


def test_unusable_rows_are_skipped(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"

    write_jsonl(
        path,
        [
            {"unknown": "ignored"},
            {"prompt": "valid"},
        ],
    )

    examples = JSONLDatasetSource(path).load()

    assert len(examples) == 1
    assert examples[0].input == "valid"


def test_non_object_json_row_raises_value_error(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        JSONLDatasetSource(path).load()


def test_invalid_json_raises_decode_error(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        JSONLDatasetSource(path).load()
