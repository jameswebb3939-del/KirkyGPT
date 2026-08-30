from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from src.llm_followups.persistence.models import (
    Base,
)


def project_root() -> Path:
    """
    Return the root directory of the project.
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """
    Return the project data directory.
    """
    return project_root() / "data"


@pytest.fixture(scope="session")
def sft_jsonl_path(
    data_dir: Path,
) -> Path:
    """
    Return the SFT JSONL file.
    """
    path = (
        data_dir
        / "sft_followups.jsonl"
    )

    if (
        not path.exists()
        or not path.is_file()
    ):
        raise FileNotFoundError(
            f"SFT JSONL file not found: {path}"
        )

    return path


def iter_jsonl(
    path: Path,
) -> Iterator[dict[str, Any]]:
    """
    Iterate over JSONL objects.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for lineno, raw in enumerate(
            file,
            1,
        ):
            line = raw.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on "
                    f"line {lineno}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {lineno}: expected "
                    "JSON object, got "
                    f"{type(obj).__name__}"
                )

            yield obj


@pytest.fixture(scope="session")
def sft_rows(
    sft_jsonl_path: Path,
) -> list[dict[str, Any]]:
    """
    Return all SFT rows.
    """
    rows = list(
        iter_jsonl(sft_jsonl_path)
    )

    if not rows:
        raise ValueError(
            "No rows found in "
            f"{sft_jsonl_path}"
        )

    return rows


# ==========================================================
# SQLite integration-test fixtures
# ==========================================================


@pytest_asyncio.fixture
async def sqlite_session_factory(
) -> AsyncIterator[
    async_sessionmaker[AsyncSession]
]:
    """
    Create an isolated SQLite database for
    each integration test.

    StaticPool keeps all operations on the
    same in-memory SQLite database.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        echo=False,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
    )

    @event.listens_for(
        engine.sync_engine,
        "connect",
    )
    def enable_foreign_keys(
        dbapi_connection,
        connection_record,
    ) -> None:
        del connection_record

        cursor = (
            dbapi_connection.cursor()
        )

        cursor.execute(
            "PRAGMA foreign_keys=ON"
        )

        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all
        )

    session_factory = (
        async_sessionmaker[
            AsyncSession
        ](
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )
    )

    try:
        yield session_factory

    finally:
        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.drop_all
            )

        await engine.dispose()