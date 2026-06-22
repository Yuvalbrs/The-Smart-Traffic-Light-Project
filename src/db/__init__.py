"""T-01-03 - SQLite (WAL) persistence layer.

ADR-001 storage split: SQLite (WAL) holds metadata, the provenance/version
chain, and per-episode KPIs; bulk per-vehicle traces live in JSONL (T-01-04),
trip KPIs in SUMO ``tripinfo.xml``. ``specs/data-schema.md`` (vault) is the SSOT
for every field name, type, and unit here.

Public surface:

- :mod:`src.db.models` - the SQLAlchemy ORM models (``Base`` + 6 tables).
- :func:`src.db.engine.create_db_engine` - an engine with the WAL PRAGMAs wired
  onto every connection.
- :func:`src.db.engine.init_db` - create all tables (the migration entry point;
  also exposed as ``python -m scripts.init_db``).
"""

from __future__ import annotations

from src.db.engine import create_db_engine, init_db
from src.db.models import (
    Base,
    Episode,
    EpisodeKpi,
    ExperimentRun,
    ModelArtifact,
    Observation,
    VehicleSnapshot,
)

__all__ = [
    "Base",
    "Episode",
    "EpisodeKpi",
    "ExperimentRun",
    "ModelArtifact",
    "Observation",
    "VehicleSnapshot",
    "create_db_engine",
    "init_db",
]
