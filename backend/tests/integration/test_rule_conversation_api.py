from __future__ import annotations

import httpx
import pytest

from llm_followups.persistence.unit_of_work import (
    UnitOfWork,
)
from llm_followups.rules.runtime import (
    RuleRuntime,
)
from llm_followups.server.main import (
    create_app,
)
from llm_followups.services.chat_history import (
    ChatHistoryService,
)
from llm_followups.utils.config import (
    Settings,
)


pytestmark = pytest.mark.integration


async def no_database_startup() -> None:
    return None


def make_rule_app(
    sqlite_session_factory,
    runtime: RuleRuntime,
):
    service = ChatHistoryService(
        runtime,
        uow_factory=lambda: UnitOfWork(
            sqlite_session_factory
        ),
    )

    return create_app(
        Settings(device="cpu"),
        runtime=runtime,
        chat_history=service,
        init_database=no_database_startup,
    )


@pytest.mark.asyncio
async def test_rule_conversation_survives_service_restart(
    sqlite_session_factory,
) -> None:
    """
    Verify rule state is reconstructed from persisted SQLite history,
    rather than being held only in process memory.
    """

    first_runtime = RuleRuntime(
        Settings(device="cpu")
    )
    await first_runtime.load()

    first_app = make_rule_app(
        sqlite_session_factory,
        first_runtime,
    )

    first_transport = httpx.ASGITransport(
        app=first_app
    )

    async with httpx.AsyncClient(
        transport=first_transport,
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/conversations",
            json={"title": "New chat"},
        )

        assert created.status_code == 201

        conversation_id = (
            created.json()["id"]
        )

        first_turn = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": "Help me with Kirk"
            },
        )

        assert first_turn.status_code == 200

        messages = (
            first_turn.json()["messages"]
        )

        assert messages[-1]["content"] == (
            "Are you mourning Charlie for the Kirkiversary, "
            "hunting the real killers, or both?"
        )

    # Simulate a backend/service restart by constructing an entirely
    # new runtime and service while reusing the same SQLite database.
    second_runtime = RuleRuntime(
        Settings(device="cpu")
    )
    await second_runtime.load()

    second_app = make_rule_app(
        sqlite_session_factory,
        second_runtime,
    )

    second_transport = httpx.ASGITransport(
        app=second_app
    )

    async with httpx.AsyncClient(
        transport=second_transport,
        base_url="http://test",
    ) as client:
        second_turn = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": "hunt"
            },
        )

        assert second_turn.status_code == 200

        messages = (
            second_turn.json()["messages"]
        )

        assert (
            "For the real hunt, start with Erika's sudden widow glow-up"
            in messages[-1]["content"]
        )

        assert (
            "Do you need help with Erika theories, "
            "shooter motives, or Kirkiversary memes?"
            in messages[-1]["content"]
        )


@pytest.mark.asyncio
async def test_persistent_conversation_can_switch_topics(
    sqlite_session_factory,
) -> None:
    runtime = RuleRuntime(
        Settings(device="cpu")
    )
    await runtime.load()

    app = make_rule_app(
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
        created = await client.post(
            "/conversations",
            json={"title": "New chat"},
        )

        conversation_id = (
            created.json()["id"]
        )

        first = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": "Help me with Kirk"
            },
        )

        assert first.status_code == 200

        switched = await client.post(
            (
                "/conversations/"
                f"{conversation_id}"
                "/chat"
            ),
            json={
                "content": (
                    "Actually help me with conspiracy"
                )
            },
        )

        assert switched.status_code == 200

        assert (
            switched.json()["messages"][-1]
            ["content"]
            == (
                "Are you mapping the cover-up, "
                "the beneficiaries, or the next target?"
            )
        )