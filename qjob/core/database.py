from __future__ import annotations

import contextlib
import os
import typing

import sqlalchemy
import sqlalchemy.engine
import sqlalchemy.orm
import sqlalchemy.pool

import qjob.core.models as models

# --------------------------------------------------------------------------------------
# Module-level engine and session factory
#
# Both are initialised lazily by calling ``init_db()``.  All other modules must
# call ``get_session()`` rather than constructing sessions directly.

_engine:         sqlalchemy.engine.Engine | None = None
_SessionFactory: sqlalchemy.orm.sessionmaker | None = None

# --------------------------------------------------------------------------------------
# Dialect helpers

_IS_POSTGRES_PREFIXES = ("postgresql", "postgres")
_POOL_ENABLED_ENV = "QJOB_DB_POOL_ENABLED"
_POOL_SIZE_ENV = "QJOB_DB_POOL_SIZE"
_MAX_OVERFLOW_ENV = "QJOB_DB_MAX_OVERFLOW"


def _is_postgres(url: str) -> bool:
    """Return True when *url* targets a PostgreSQL database."""
    return any(url.startswith(p) for p in _IS_POSTGRES_PREFIXES)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    """Parse a positive integer environment variable."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.")
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0, got {value}.")
    return value


# --------------------------------------------------------------------------------------
# Public API


def init_db(url: str | None = None) -> sqlalchemy.engine.Engine:
    """
    Initialise the PostgreSQL database engine.

    This function is idempotent: calling it multiple times with the same URL
    has no effect after the first call.

    The engine uses ``NullPool`` by default so multi-process API deployments
    do not multiply idle database connections. Set ``QJOB_DB_POOL_ENABLED=1``
    to opt into a per-process QueuePool. Schema creation and upgrades are
    handled by Alembic, not by this function.

    Parameters
    ----------
    url : str | None
        SQLAlchemy database URL.  When *None*, ``QJOB_DB_URL`` is read from
        the environment.

    Returns
    -------
    sqlalchemy.engine.Engine
        The initialised engine.

    Raises
    ------
    RuntimeError
        If no database URL is configured, a non-PostgreSQL URL is supplied,
        or called a second time with a *different* URL than the first call.
    """

    global _engine, _SessionFactory

    resolved_url = url or os.environ.get("QJOB_DB_URL")
    if not resolved_url:
        raise RuntimeError(
            "QJOB_DB_URL must be set to a PostgreSQL database URL. "
            "Example: postgresql+psycopg://qjob:password@localhost:5432/qjob"
        )
    if not _is_postgres(resolved_url):
        raise RuntimeError(
            f"Only PostgreSQL database URLs are supported; got {resolved_url!r}."
        )

    if _engine is not None:
        existing = _engine.url
        requested = sqlalchemy.engine.make_url(resolved_url)
        if existing != requested:
            raise RuntimeError(
                f"init_db() already called with url={_engine.url!r}; "
                f"cannot reinitialise with url={resolved_url!r}."
            )
        return _engine

    engine_kwargs = _engine_kwargs()

    _engine = sqlalchemy.create_engine(resolved_url, **engine_kwargs)
    _SessionFactory = sqlalchemy.orm.sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    # Insert the default resource row if the migrated table is empty.
    _ensure_default_resource()

    return _engine


def _engine_kwargs() -> dict:
    """Return SQLAlchemy engine options derived from environment settings."""

    kwargs: dict = {
        "echo": False,
        "pool_pre_ping": True,
    }
    if _env_bool(_POOL_ENABLED_ENV, default=False):
        kwargs["pool_size"] = _env_positive_int(_POOL_SIZE_ENV, 5)
        kwargs["max_overflow"] = _env_positive_int(_MAX_OVERFLOW_ENV, 5)
    else:
        kwargs["poolclass"] = sqlalchemy.pool.NullPool
    return kwargs


def get_engine() -> sqlalchemy.engine.Engine:
    """
    Return the active engine.

    Parameters
    ----------
    None

    Returns
    -------
    sqlalchemy.engine.Engine
        The engine created by ``init_db()``.

    Raises
    ------
    RuntimeError
        If ``init_db()`` has not been called yet.
    """

    if _engine is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")
    return _engine


@contextlib.contextmanager
def get_session() -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """
    Yield a transactional database session, committing on success and
    rolling back on any exception.

    Parameters
    ----------
    None

    Yields
    ------
    sqlalchemy.orm.Session
        An open session bound to the engine created by ``init_db()``.

    Raises
    ------
    RuntimeError
        If ``init_db()`` has not been called yet.

    Examples
    --------
    >>> with database.get_session() as session:
    ...     session.add(job)
    """

    if _SessionFactory is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")

    session: sqlalchemy.orm.Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextlib.contextmanager
def get_session_for_update() -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """
    Yield a session that locks the ``resources`` row for atomic update.

    This issues ``SELECT ... FOR UPDATE`` so concurrent schedulers cannot read
    stale resource counts.

    Parameters
    ----------
    None

    Yields
    ------
    sqlalchemy.orm.Session
        An open session with the resources row already locked.

    Raises
    ------
    RuntimeError
        If ``init_db()`` has not been called yet.
    """

    if _SessionFactory is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")

    session: sqlalchemy.orm.Session = _SessionFactory()
    try:
        # Lock the single resources row for the duration of the transaction.
        session.execute(
            sqlalchemy.text(
                "SELECT id FROM resources WHERE id = 1 FOR UPDATE"
            )
        )
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """
    Dispose the active engine and reset module-level state.

    This function is intended for tests that need to simulate an uninitialised
    process. It deliberately does not drop tables; schema lifecycle is managed
    by Alembic.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global _engine, _SessionFactory

    if _engine is not None:
        _engine.dispose()

    _engine = None
    _SessionFactory = None


# --------------------------------------------------------------------------------------
# Private helpers


def _ensure_default_resource() -> None:
    """Insert a default Resource row (id=1) if none exists."""

    if _SessionFactory is None:
        return

    session: sqlalchemy.orm.Session = _SessionFactory()
    try:
        exists = session.get(models.Resource, 1)
        if exists is None:
            session.add(models.Resource(id=1))
            session.commit()
    finally:
        session.close()
