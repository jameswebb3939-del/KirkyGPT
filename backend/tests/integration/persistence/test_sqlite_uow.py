from __future__ import annotations

import pytest

from sqlalchemy.exc import (
    IntegrityError,
)

from src.llm_followups.persistence.models import (
    ConversationModel,
    MessageModel,
)
from src.llm_followups.persistence.unit_of_work import (
    UnitOfWork,
)


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_repository_round_trip(
    sqlite_session_factory,
) -> None:
    conversation = ConversationModel(
        title="Docker chat"
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        await uow.conversations.add(
            conversation
        )

        await uow.commit()

    conversation_id = (
        conversation.id
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        loaded = (
            await uow.conversations.get(
                conversation_id
            )
        )

    assert loaded is not None
    assert loaded.id == conversation_id

    assert (
        loaded.title
        == "Docker chat"
    )


@pytest.mark.asyncio
async def test_message_repository_preserves_order(
    sqlite_session_factory,
) -> None:
    conversation = ConversationModel(
        title="Ordering test"
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        await uow.conversations.add(
            conversation
        )

        await uow.commit()

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert uow.messages is not None

        await uow.messages.add(
            MessageModel(
                conversation_id=(
                    conversation.id
                ),
                role="assistant",
                content="Second",
                position=1,
            )
        )

        await uow.messages.add(
            MessageModel(
                conversation_id=(
                    conversation.id
                ),
                role="user",
                content="First",
                position=0,
            )
        )

        await uow.commit()

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert uow.messages is not None

        messages = (
            await uow.messages
            .list_for_conversation(
                conversation.id
            )
        )

    assert [
        message.content
        for message in messages
    ] == [
        "First",
        "Second",
    ]


@pytest.mark.asyncio
async def test_uncommitted_work_is_rolled_back(
    sqlite_session_factory,
) -> None:
    conversation = ConversationModel(
        title="Should rollback"
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        await uow.conversations.add(
            conversation
        )

        # Deliberately no commit.

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        conversations = (
            await uow.conversations.list_all()
        )

    assert conversations == []


@pytest.mark.asyncio
async def test_failed_commit_rolls_back_entire_message_pair(
    sqlite_session_factory,
) -> None:
    """
    Prove atomicity.

    Both pending INSERTs use the same
    (conversation_id, position), violating
    the UNIQUE constraint.

    The transaction must persist neither.
    """
    conversation = ConversationModel(
        title="Atomic transaction"
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        await uow.conversations.add(
            conversation
        )

        await uow.commit()

    with pytest.raises(
        IntegrityError
    ):
        async with UnitOfWork(
            sqlite_session_factory
        ) as uow:
            assert (
                uow.messages
                is not None
            )

            await uow.messages.add(
                MessageModel(
                    conversation_id=(
                        conversation.id
                    ),
                    role="user",
                    content="User",
                    position=0,
                )
            )

            # Same position deliberately
            # breaks the unique constraint.
            await uow.messages.add(
                MessageModel(
                    conversation_id=(
                        conversation.id
                    ),
                    role="assistant",
                    content="Assistant",
                    position=0,
                )
            )

            await uow.commit()

    # New transaction proves nothing
    # survived the failed transaction.
    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert uow.messages is not None

        messages = (
            await uow.messages
            .list_for_conversation(
                conversation.id
            )
        )

    assert messages == []


@pytest.mark.asyncio
async def test_delete_conversation_cascades_messages(
    sqlite_session_factory,
) -> None:
    conversation = ConversationModel(
        title="Delete test"
    )

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        await uow.conversations.add(
            conversation
        )

        await uow.commit()

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert uow.messages is not None

        await uow.messages.add(
            MessageModel(
                conversation_id=(
                    conversation.id
                ),
                role="user",
                content="Hello",
                position=0,
            )
        )

        await uow.commit()

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert (
            uow.conversations
            is not None
        )

        deleted = (
            await uow.conversations.delete(
                conversation.id
            )
        )

        assert deleted is True

        await uow.commit()

    async with UnitOfWork(
        sqlite_session_factory
    ) as uow:
        assert uow.messages is not None

        messages = (
            await uow.messages
            .list_for_conversation(
                conversation.id
            )
        )

    assert messages == []