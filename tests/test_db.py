"""T-01-03 - Tests for the SQLite (WAL) schema + engine.

Covers the DoD: migration runs cleanly, WAL is active, the provenance/version
columns exist, and the ``episode`` table carries the B3 gridlock-guard columns.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.engine import create_db_engine, init_db
from src.db.models import (
    Episode,
    EpisodeKpi,
    ExperimentRun,
    Observation,
)
from src.schema.validate import SCHEMA_VERSION


@pytest.fixture()
def engine(tmp_path):
    """A file-backed engine with all tables created (WAL needs a real file)."""
    eng = create_db_engine(tmp_path / "test.db")
    init_db(eng)
    return eng


def _make_run(**overrides) -> ExperimentRun:
    base = dict(
        name="r0", mode="training", controller="dqn", config={"lr": 1e-4}
    )
    base.update(overrides)
    return ExperimentRun(**base)


def test_init_db_creates_all_tables(engine):
    tables = set(inspect(engine).get_table_names())
    assert tables == {
        "experiment_run",
        "episode",
        "observation",
        "episode_kpi",
        "model_artifact",
        "vehicle_snapshot",
    }


def test_wal_mode_is_active(engine):
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode;")).scalar()
    assert mode.lower() == "wal"


def test_foreign_keys_enforced(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys;")).scalar() == 1


def test_init_db_is_idempotent(engine):
    # Running the migration again must not raise and must not duplicate tables.
    init_db(engine)
    assert "episode" in inspect(engine).get_table_names()


def test_episode_has_gridlock_guard_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("episode")}
    assert {
        "loaded_count",
        "departed_count",
        "arrived_count",
        "insertion_backlog_fraction",
        "gridlock_censored",
    } <= cols


def test_experiment_run_has_version_chain_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("experiment_run")}
    assert {"data_version", "lstm_version", "run_id", "git_sha", "sumo_version"} <= cols


def test_schema_version_defaults_to_current(engine):
    with Session(engine) as s:
        run = _make_run()
        s.add(run)
        s.flush()
        ep = Episode(run_id_fk=run.id, index_in_run=0, seed=42, scenario="SCN-01")
        s.add(ep)
        s.commit()
        assert run.schema_version == SCHEMA_VERSION
        assert ep.schema_version == SCHEMA_VERSION
        # gridlock_censored defaults to False, not NULL.
        assert ep.gridlock_censored is False


def test_full_round_trip_with_relationships(engine):
    with Session(engine) as s:
        run = _make_run(data_version="d1", run_id="uuid-1")
        ep = Episode(
            run=run,
            index_in_run=0,
            seed=42,
            scenario="SCN-01",
            total_reward=-12.5,
            sim_duration=3600.0,
            done_reason="time_limit",
            loaded_count=800,
            departed_count=796,
            arrived_count=786,
            insertion_backlog_fraction=0.005,
            gridlock_censored=False,
        )
        ep.observations.append(
            Observation(step=0, sim_time=0.0, state={"v": [0.0]}, action=3, reward=-1.2)
        )
        ep.kpi = EpisodeKpi(avg_waiting_time=4.2, per_movement_max_wait=[1.0] * 12)
        s.add(run)
        s.commit()
        run_id = run.id

    with Session(engine) as s:
        loaded = s.get(ExperimentRun, run_id)
        assert loaded.run_id == "uuid-1"
        assert len(loaded.episodes) == 1
        ep = loaded.episodes[0]
        assert ep.departed_count == 796 and ep.arrived_count == 786
        assert ep.observations[0].action == 3
        assert ep.kpi.per_movement_max_wait == [1.0] * 12


def test_episode_kpi_is_one_to_one(engine):
    """The unique constraint on episode_kpi.episode_id_fk enforces 1:1."""
    with Session(engine) as s:
        run = _make_run()
        ep = Episode(run=run, index_in_run=0, seed=1, scenario="SCN-01")
        s.add_all([run, ep])
        s.flush()
        s.add(EpisodeKpi(episode_id_fk=ep.id, avg_waiting_time=1.0))
        s.commit()
        s.add(EpisodeKpi(episode_id_fk=ep.id, avg_waiting_time=2.0))
        with pytest.raises(IntegrityError):
            s.commit()
