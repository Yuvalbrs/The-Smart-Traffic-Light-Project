"""T-01-03 - SQLite engine factory with the WAL PRAGMAs wired per-connection.

``journal_mode=WAL`` persists in the DB header, but ``synchronous``,
``busy_timeout`` and ``foreign_keys`` are **per-connection** and must be
re-applied on every connect - so they go through a SQLAlchemy ``connect`` event
(research-systems.md s3). The WAL single-writer rule (sqlite.org/wal.html) means
all writes must be funnelled through one connection/task downstream; this module
only sets the engine up correctly.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from src.db.models import Base


def create_db_engine(db_path: str | Path, *, echo: bool = False) -> Engine:
    """Create a SQLite engine with WAL + the safe per-connection PRAGMAs.

    Parameters
    ----------
    db_path : str or Path
        Filesystem path for the SQLite database file. Use ``":memory:"`` for an
        in-memory database (tests).
    echo : bool, optional
        If ``True``, log emitted SQL. Default ``False``.

    Returns
    -------
    sqlalchemy.Engine
        An engine that applies ``journal_mode=WAL``, ``synchronous=NORMAL``,
        ``busy_timeout=5000`` and ``foreign_keys=ON`` to every connection.
    """
    url = "sqlite:///:memory:" if str(db_path) == ":memory:" else f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        echo=echo,
        connect_args={"check_same_thread": False},  # allow cross-thread access
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):  # noqa: ANN001, ANN202
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")  # persistent, cheap to re-assert
        cur.execute("PRAGMA synchronous=NORMAL;")  # WAL-safe; durable across app crash
        cur.execute("PRAGMA busy_timeout=5000;")  # wait up to 5s for the write lock
        cur.execute("PRAGMA foreign_keys=ON;")  # SQLite defaults OFF
        cur.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create every table defined on :class:`src.db.models.Base` if absent.

    This is the migration entry point. ``create_all`` is idempotent - it only
    issues ``CREATE TABLE`` for tables that do not yet exist, so re-running it on
    an existing database is a no-op.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        The target engine (typically from :func:`create_db_engine`).
    """
    Base.metadata.create_all(engine)
