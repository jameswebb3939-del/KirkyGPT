from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from src.llm_followups.persistence.unit_of_work import (
    UnitOfWork,
)
from src.llm_followups.server.main import (
    create_app,
)
from src.llm_followups.services.chat_history import (
    ChatHistoryService,
)
from src.llm_followups.utils.config import (
    Settings,
)


pytestmark = pytest.mark.integration


ASSISTANT_RESPONSE = (
    "- What environment are you using?\n"
    "- What constraints should I consider?\n"
    "- What outcome do you want?"
)


class DummyRuntime:
    def __init__(
        self,
        *,
        fail_generation: bool = False,
    ) -> None:
        self.fail_generation = (
            fail_generation
        )

        self.received_messages = []

    async def load(
        self,
    ) -> None:
        return None

    def is_loaded(
        self,
    ) -> bool:
        return True

    def model_name(
        self,
    ) -> str:
        return "dummy-model"

    def device_str(
        self,
    ) -> str:
        return "cpu"

    def adapter_loaded(
        self,
    ) -> bool:
        return False

    def make_request(
        self,
        messages,
        **kwargs,
    ):
        del kwargs

        self.received_messages = list(
            messages
        )

        return {
            "messages": messages
        }

    async def generate(
        self,
        request,
    ):
        del request

        if self.fail_generation:
            raise RuntimeError(
                "dummy generation failure"
            )

        return SimpleNamespace(
            raw_text=ASSISTANT_RESPONSE,
            final_text=ASSISTANT_RESPONSE,
            used_repair=False,
            used_fallback=False,
            latency_ms=1,
        )


async def no_database_startup(
) -> None:
    return None


def make_app(
    sqlite_session_factory,
    runtime: DummyRuntime,
):
    settings = Settings(
        model_name="dummy-model",
        device="cpu",
    )

    service = ChatHistoryService(
        runtime,
        uow_factory=lambda: UnitOfWork(
            sqlite_session_factory
        ),
    )

    return create_app(
        settings,
        runtime=runtime,
        chat_history=service,
        init_database=(
            no_database_startup
        ),
    )


@pytest.mark.asyncio
async def test_conversation_api_full_lifecycle(
    sqlite_session_factory,
) -> None:
    runtime = DummyRuntime()

    app = make_app(
        sqlite_session_factory,
        runtime,
    )

    transport = httpx.ASGITransport(
        app=app
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        # -------------------------
        # Create
        # -------------------------

        response = await client.post(
            "/conversations",
            json={
                "title": "New chat"
            },
        )

        assert (
            response.status_code
            == 201
        )

        created = response.json()

        conversation_id = (
            created["id"]
        )

        assert (
            created["title"]
            == "New chat"
        )

        assert (
            created["messages"]
            == []
        )

        # -------------------------
        # Send message
        # -------------------------

        response = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": (
                    "Explain Docker"
                )
            },
        )

        assert (
            response.status_code
            == 200
        )

        updated = response.json()

        assert (
            updated["title"]
            == "Explain Docker"
        )

        assert len(
            updated["messages"]
        ) == 2

        assert [
            message["role"]
            for message
            in updated["messages"]
        ] == [
            "user",
            "assistant",
        ]

        assert [
            message["position"]
            for message
            in updated["messages"]
        ] == [
            0,
            1,
        ]

        assert (
            updated["messages"][0]
            ["content"]
            == "Explain Docker"
        )

        assert (
            updated["messages"][1]
            ["content"]
            == ASSISTANT_RESPONSE
        )

        # -------------------------
        # List
        # -------------------------

        response = await client.get(
            "/conversations"
        )

        assert (
            response.status_code
            == 200
        )

        history = response.json()

        assert len(history) == 1

        assert (
            history[0]["id"]
            == conversation_id
        )

        # -------------------------
        # Get
        # -------------------------

        response = await client.get(
            "/conversations/"
            f"{conversation_id}"
        )

        assert (
            response.status_code
            == 200
        )

        loaded = response.json()

        assert len(
            loaded["messages"]
        ) == 2

        # -------------------------
        # Delete
        # -------------------------

        response = await client.delete(
            "/conversations/"
            f"{conversation_id}"
        )

        assert (
            response.status_code
            == 204
        )

        # -------------------------
        # Confirm deleted
        # -------------------------

        response = await client.get(
            "/conversations/"
            f"{conversation_id}"
        )

        assert (
            response.status_code
            == 404
        )


@pytest.mark.asyncio
async def test_generation_failure_does_not_persist_user_message(
    sqlite_session_factory,
) -> None:
    runtime = DummyRuntime(
        fail_generation=True
    )

    app = make_app(
        sqlite_session_factory,
        runtime,
    )

    transport = httpx.ASGITransport(
        app=app
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        create_response = (
            await client.post(
                "/conversations",
                json={
                    "title": "New chat"
                },
            )
        )

        conversation_id = (
            create_response.json()[
                "id"
            ]
        )

        response = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": (
                    "Explain Docker"
                )
            },
        )

        assert (
            response.status_code
            == 500
        )

        # The failed LLM interaction
        # must not leave a dangling
        # user message in SQLite.
        response = await client.get(
            "/conversations/"
            f"{conversation_id}"
        )

        assert (
            response.status_code
            == 200
        )

        conversation = (
            response.json()
        )

        assert (
            conversation["messages"]
            == []
        )


@pytest.mark.asyncio
async def test_missing_conversation_returns_404(
    sqlite_session_factory,
) -> None:
    runtime = DummyRuntime()

    app = make_app(
        sqlite_session_factory,
        runtime,
    )

    transport = httpx.ASGITransport(
        app=app
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/conversations/missing"
        )

    assert response.status_code == 404