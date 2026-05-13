# tests/conftest.py

import datetime
import os

import pytest
import sqlalchemy

import qjob.core.database as database
import qjob.core.models as models

_POSTGRES_PREFIXES = ("postgresql", "postgres")


def _test_db_url() -> str:
    """Return the PostgreSQL database URL used by the test suite."""

    url = os.environ.get("QJOB_TEST_DB_URL")
    if not url:
        raise RuntimeError(
            "QJOB_TEST_DB_URL must be set to a migrated PostgreSQL test database. "
            "Refusing to use QJOB_DB_URL because tests delete table data."
        )
    if not url.startswith(_POSTGRES_PREFIXES):
        raise RuntimeError(f"QJOB_TEST_DB_URL must be PostgreSQL; got {url!r}.")
    return url


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """
    Initialise and clear a PostgreSQL test database for each test.

    The database schema must already exist via ``alembic upgrade head``.
    Tests delete table data but do not drop tables.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set QJOB_DB_URL so CLI commands and the API server both
        connect to the test database.

    Yields
    ------
    None
    """

    url = _test_db_url()
    monkeypatch.setenv("QJOB_DB_URL", url)
    database.init_db(url)
    _clear_data()
    yield
    try:
        database.init_db(url)
        _clear_data()
    finally:
        database.reset_db()


def _clear_data() -> None:
    """Delete test data while preserving the migrated schema."""

    with database.get_engine().begin() as connection:
        connection.execute(sqlalchemy.delete(models.Job))
        connection.execute(sqlalchemy.delete(models.Resource))

    with database.get_session() as session:
        session.add(models.Resource(id=1))


def as_utc(dt: datetime.datetime) -> datetime.datetime:
    """Attach UTC tzinfo to a naive datetime."""

    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt
