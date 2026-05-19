# tests/conftest.py

import datetime
import hashlib
import os
import secrets

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
        connection.execute(sqlalchemy.delete(models.ApiToken))
        connection.execute(sqlalchemy.delete(models.Job))
        connection.execute(sqlalchemy.delete(models.Resource))

    with database.get_session() as session:
        session.add(models.Resource(id=1))


def _insert_token(username: str) -> str:
    """Insert an API token for *username* directly into the DB and return the raw token."""
    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with database.get_session() as session:
        session.add(models.ApiToken(username=username, token_hash=token_hash))
    return token


@pytest.fixture
def alice_token():
    """Raw API token for 'alice'."""
    return _insert_token("alice")


@pytest.fixture
def bob_token():
    """Raw API token for 'bob'."""
    return _insert_token("bob")


@pytest.fixture
def root_token():
    """Raw API token for 'root' (admin via QJOB_ADMIN_USERS default)."""
    return _insert_token("root")


def as_utc(dt: datetime.datetime) -> datetime.datetime:
    """Attach UTC tzinfo to a naive datetime."""

    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt
