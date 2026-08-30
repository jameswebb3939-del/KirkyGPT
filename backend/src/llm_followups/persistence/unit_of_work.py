from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from llm_followups.persistence.database import (
    SessionFactory,
)
from llm_followups.persistence.repositories import (
    ConversationRepository,
    MessageRepository,
)


class UnitOfWork:
    """
    Own one SQLAlchemy session and transaction boundary.

    Repositories never commit independently.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[
            AsyncSession
        ] = SessionFactory,
    ) -> None:
        self._session_factory = (
            session_factory
        )

        self.session: AsyncSession | None = (
            None
        )

        self.conversations: (
            ConversationRepository | None
        ) = None

        self.messages: (
            MessageRepository | None
        ) = None

        self._committed = False

    async def __aenter__(
        self,
    ) -> "UnitOfWork":
        self.session = (
            self._session_factory()
        )

        self.conversations = (
            ConversationRepository(
                self.session
            )
        )

        self.messages = (
            MessageRepository(
                self.session
            )
        )

        self._committed = False

        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError(
                "UnitOfWork has not been entered"
            )

        await self.session.commit()
        self._committed = True

    async def rollback(self) -> None:
        if self.session is None:
            return

        await self.session.rollback()

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        del exc, traceback

        if self.session is None:
            return

        try:
            if exc_type is not None:
                await self.rollback()

            elif not self._committed:
                # Safe default:
                # uncommitted work does not leak.
                await self.rollback()

        finally:
            await self.session.close()