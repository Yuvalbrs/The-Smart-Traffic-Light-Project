"""T-01-03 - Initialize (migrate) the SQLite results database.

Creates the SQLite file with WAL mode and every table in
``specs/data-schema.md`` (vault). Idempotent: re-running it leaves an existing
database untouched (``create_all`` only creates missing tables).

Run::

    python -m scripts.init_db                  # -> data/traffic.db
    python -m scripts.init_db --db results.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import inspect

from src.db.engine import create_db_engine, init_db

_DEFAULT_DB = Path("data") / "traffic.db"


def main() -> None:
    """Parse args, create the database + tables, and report what exists."""
    parser = argparse.ArgumentParser(description="Create/migrate the SQLite results DB.")
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"path to the SQLite file (default: {_DEFAULT_DB})",
    )
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(args.db)
    added = init_db(engine)
    if added:
        print(f"[init_db] migrated - added columns: {', '.join(added)}")

    tables = sorted(inspect(engine).get_table_names())
    journal_mode = engine.raw_connection().driver_connection.execute(
        "PRAGMA journal_mode;"
    ).fetchone()[0]
    print(f"[init_db] {args.db} ready (journal_mode={journal_mode})")
    print(f"[init_db] tables: {', '.join(tables)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
