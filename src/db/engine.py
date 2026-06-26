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

from sqlalchemy import Engine, create_engine, event, inspect, text

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


def _add_missing_columns(engine: Engine) -> list[str]:
    """Idempotently ALTER existing tables to add any model column they lack.

    ``create_all`` makes missing *tables* but never alters an existing one, so a
    schema that GREW (e.g. the E5 ``per_movement_p95_wait`` / ``worst_movement_max_wait``
    KPI columns added for T-04) would be silently missing on a database created by an
    earlier version. SQLite supports ``ALTER TABLE ... ADD COLUMN``; every column added
    this way must be nullable (so existing rows back-fill with NULL) - which holds for all
    the grown columns. Returns the ``table.column`` names added, for logging.
    """
    insp = inspect(engine)
    live_tables = set(insp.get_table_names())
    added: list[str] = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in live_tables:
                continue  # create_all already made it with every column
            live_cols = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in live_cols:
                    continue
                ddl_type = col.type.compile(engine.dialect)
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl_type}')
                )
                added.append(f"{table.name}.{col.name}")
    return added


def init_db(engine: Engine) -> list[str]:
    """Create missing tables AND add missing columns to existing ones (idempotent migration).

    Two passes: ``create_all`` issues ``CREATE TABLE`` only for absent tables, then
    :func:`_add_missing_columns` ALTERs any pre-existing table to add columns the model
    grew since it was created. Re-running on an up-to-date database is a no-op. Returns the
    list of ``table.column`` names added (empty when nothing changed).

    Parameters
    ----------
    engine : sqlalchemy.Engine
        The target engine (typically from :func:`create_db_engine`).
    """
    Base.metadata.create_all(engine)
    return _add_missing_columns(engine)
