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

    Repositories perform persistence operations but never
    commit or roll back independently.

    Behaviour:
    - commit() explicitly makes writes durable
    - exceptions cause rollback()
    - clean read-only exits simply close the session
    - closing an uncommitted SQLAlchemy session safely
      discards any outstanding transaction
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[
            AsyncSession
        ] = SessionFactory,
    ) -> None:
        self._session_factory = session_factory

        self.session: AsyncSession | None = None

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
        self.session = self._session_factory()

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

        self._committed = False

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
            # Explicit rollback is required when
            # something failed inside the UoW.
            if exc_type is not None:
                await self.rollback()

            # IMPORTANT:
            #
            # Do NOT explicitly rollback a successful
            # read-only UoW here.
            #
            # session.rollback() expires ORM objects.
            # Once the session is subsequently closed,
            # callers receive detached + expired objects,
            # causing DetachedInstanceError.
            #
            # session.close() itself safely releases /
            # rolls back an uncommitted DB transaction.

        finally:
            await self.session.close()

            self.session = None
            self.conversations = None
            self.messages = None