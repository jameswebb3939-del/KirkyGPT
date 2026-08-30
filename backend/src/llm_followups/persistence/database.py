from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from llm_followups.persistence.models import Base


DEFAULT_DB_PATH = Path("data") / "ec_pro.db"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{DEFAULT_DB_PATH.as_posix()}",
)


engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
)


# SQLite does not enable FK enforcement by default.
@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(
    dbapi_connection,
    connection_record,
) -> None:
    del connection_record

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionFactory = async_sessionmaker[
    AsyncSession
](
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Create application tables.

    For this local project create_all() is sufficient.
    A production system should eventually use migrations.
    """
    DEFAULT_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
        )


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session