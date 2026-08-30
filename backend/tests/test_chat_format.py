from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.llm_followups.server.main import (
    create_app,
)
from src.llm_followups.server.schemas import (
    ChatRequest,
)
from src.llm_followups.tuning.validate import (
    validate_followup_list,
)
from src.llm_followups.utils.config import (
    Settings,
)


class DummyResult:
    def __init__(
        self,
        text: str,
    ) -> None:
        self.raw_text = text
        self.final_text = text
        self.used_fallback = False
        self.used_repair = False
        self.latency_ms = 1


class DummyRuntime:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._settings = settings

    async def load(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def model_name(self) -> str:
        return self._settings.model_name

    def device_str(self) -> str:
        return "cpu"

    def adapter_loaded(self) -> bool:
        return False

    def make_request(
        self,
        messages,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
    ):
        return {
            "messages": messages,
            "max_new_tokens": (
                max_new_tokens
            ),
            "temperature": temperature,
            "top_p": top_p,
        }

    async def generate(
        self,
        req,
    ):
        del req

        text = (
            "- What is your main goal "
            "with this setup?\n"
            "- What constraints should "
            "I consider?\n"
            "- What output format do "
            "you want?\n"
        )

        return DummyResult(text)


async def no_database_startup() -> None:
    return None


def make_test_client() -> TestClient:
    settings = Settings(
        model_name=(
            "meta-llama/"
            "Llama-3.2-1B-Instruct"
        ),
        device="cpu",
    )

    runtime = DummyRuntime(settings)

    app = create_app(
        settings,
        runtime=runtime,
        init_database=no_database_startup,
    )

    return TestClient(app)


def test_health_endpoint() -> None:
    client = make_test_client()

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert (
        body["model_loaded"]
        is True
    )

    assert body["model_name"] == (
        "meta-llama/"
        "Llama-3.2-1B-Instruct"
    )

    assert body["device"] == "cpu"


def test_chat_endpoint_returns_followup_bullets(
) -> None:
    client = make_test_client()

    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Help me deploy "
                        "a model"
                    ),
                }
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "response_text" in body

    result = validate_followup_list(
        text=body["response_text"],
        min_questions=3,
        bullet_style="either",
        require_question_mark=True,
        forbid_extra_text=True,
    )

    assert result.ok, result.errors


def get_last_assistant_text(
    messages: list[
        dict[str, Any]
    ],
) -> str | None:
    for message in reversed(messages):
        if not isinstance(
            message,
            dict,
        ):
            continue

        if (
            message.get("role")
            != "assistant"
        ):
            continue

        content = message.get(
            "content"
        )

        if (
            isinstance(content, str)
            and content.strip()
        ):
            return content.strip()

    return None


def test_jsonl_rows_are_valid_json(
    sft_rows: list[
        dict[str, Any]
    ],
) -> None:
    assert isinstance(
        sft_rows,
        list,
    )

    assert len(sft_rows) > 0

    for row in sft_rows:
        assert isinstance(
            row,
            dict,
        )


def test_each_row_matches_chatrequest_schema(
    sft_rows: list[
        dict[str, Any]
    ],
) -> None:
    for row in sft_rows:
        try:
            ChatRequest(**row)

        except ValidationError as exc:
            raise AssertionError(
                "Schema validation failed: "
                f"{exc}"
            ) from exc


def test_assistant_messages_are_followup_bullets(
    sft_rows: list[
        dict[str, Any]
    ],
) -> None:
    for row in sft_rows:
        messages = row["messages"]

        assistant_text = (
            get_last_assistant_text(
                messages
            )
        )

        assert assistant_text is not None

        result = validate_followup_list(
            text=assistant_text,
            min_questions=3,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True,
        )

        assert result.ok, result.errors


def test_no_numbered_lists_in_assistant_output(
    sft_rows: list[
        dict[str, Any]
    ],
) -> None:
    numbered = re.compile(
        r"^\s*\d+[\.\)]\s+"
    )

    for row in sft_rows:
        assistant_text = (
            get_last_assistant_text(
                row["messages"]
            )
        )

        assert assistant_text is not None

        for line in (
            assistant_text.splitlines()
        ):
            assert not numbered.match(
                line
            ), (
                "Numbered list line "
                f"not allowed: {line!r}"
            )


def test_min_questions_respected(
    sft_rows: list[
        dict[str, Any]
    ],
) -> None:
    minimum = 3

    for row in sft_rows:
        assistant_text = (
            get_last_assistant_text(
                row["messages"]
            )
        )

        assert assistant_text is not None

        result = validate_followup_list(
            text=assistant_text,
            min_questions=minimum,
            bullet_style="either",
            require_question_mark=True,
            forbid_extra_text=True,
        )

        assert result.ok, result.errors
        assert (
            result.num_items
            >= minimum
        )