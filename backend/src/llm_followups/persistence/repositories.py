from __future__ import annotations

from sqlalchemy import (
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from llm_followups.persistence.models import (
    ConversationModel,
    MessageModel,
)


class ConversationRepository:
    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        conversation: ConversationModel,
    ) -> None:
        self._session.add(conversation)

    async def get(
        self,
        conversation_id: str,
        *,
        include_messages: bool = False,
    ) -> ConversationModel | None:
        statement = select(
            ConversationModel
        ).where(
            ConversationModel.id
            == conversation_id
        )

        if include_messages:
            statement = statement.options(
                selectinload(
                    ConversationModel.messages,
                )
            )

        result = await self._session.execute(
            statement,
        )

        return result.scalar_one_or_none()

    async def list_all(
        self,
    ) -> list[ConversationModel]:
        statement = (
            select(ConversationModel)
            .order_by(
                ConversationModel.updated_at.desc()
            )
        )

        result = await self._session.execute(
            statement,
        )

        return list(result.scalars().all())

    async def delete(
        self,
        conversation_id: str,
    ) -> bool:
        statement = (
            delete(ConversationModel)
            .where(
                ConversationModel.id
                == conversation_id
            )
        )

        result = await self._session.execute(
            statement,
        )

        return bool(result.rowcount)


class MessageRepository:
    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        message: MessageModel,
    ) -> None:
        self._session.add(message)

    async def list_for_conversation(
        self,
        conversation_id: str,
    ) -> list[MessageModel]:
        statement = (
            select(MessageModel)
            .where(
                MessageModel.conversation_id
                == conversation_id
            )
            .order_by(
                MessageModel.position.asc()
            )
        )

        result = await self._session.execute(
            statement,
        )

        return list(result.scalars().all())

    async def next_position(
        self,
        conversation_id: str,
    ) -> int:
        statement = select(
            func.max(MessageModel.position)
        ).where(
            MessageModel.conversation_id
            == conversation_id
        )

        result = await self._session.execute(
            statement,
        )

        maximum = result.scalar_one_or_none()

        if maximum is None:
            return 0

        return int(maximum) + 1