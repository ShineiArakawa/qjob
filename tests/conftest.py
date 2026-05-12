# tests/conftest.py

import datetime

import pytest

import qjob.core.database as database

# Named in-memory DB — all connections in the same process share the same data.
_TEST_DB_URL = "sqlite:///file:qjob_test?mode=memory&cache=shared&uri=true"


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """
    Initialise a shared in-memory SQLite database for each test.

    Uses a named in-memory database so that all connections within the same
    process (including the scheduler started by the FastAPI lifespan) share
    the same tables and data.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set QJOB_DB_URL so CLI commands and the API server both
        connect to the test database.

    Yields
    ------
    None
    """

    monkeypatch.setenv("QJOB_DB_URL", _TEST_DB_URL)
    database.init_db(_TEST_DB_URL)
    yield
    database.reset_db()


def as_utc(dt: datetime.datetime) -> datetime.datetime:
    """Attach UTC tzinfo to a naive datetime returned by SQLite."""

    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt
