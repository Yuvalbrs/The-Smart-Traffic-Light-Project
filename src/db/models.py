"""T-01-03 - SQLAlchemy ORM models for the SQLite (WAL) store.

The authoritative field/type/unit spec is ``specs/data-schema.md`` (vault, B2);
this module is its code realization and must not drift from it. Entity graph
(data-schema.md s1)::

    ExperimentRun 1 --- * Episode 1 --- * Observation
          |                   |        1 --- 1 EpisodeKpi
          |                   +-------- * VehicleSnapshot   (index rows; bulk in JSONL)
          +--- * ModelArtifact

Notes
-----
* ``schema_version`` is a **semver string** (``"1.1.0"``), not an int - the 1.0
  -> 1.1.0 migration changed the type (data-schema.md migration note). The value
  is sourced from :data:`src.schema.validate.SCHEMA_VERSION` so the wire schema
  and the DB never disagree.
* The integer surrogate foreign keys are named ``run_id_fk`` / ``episode_id_fk``
  to stay distinct from the UUID provenance column ``ExperimentRun.run_id``.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from src.schema.validate import SCHEMA_VERSION


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class ExperimentRun(Base):
    """One controller run (training / live / replay); root of the provenance chain.

    Carries the full version chain (``data_version``, ``lstm_version``,
    ``run_id``, ``git_sha``, ``sumo_version``) per data-schema.md s6 / hard-rule
    #7. ``lstm_version`` is null for non-hybrid (20-d base) runs.
    """

    __tablename__ = "experiment_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    name: Mapped[str] = mapped_column(String)
    mode: Mapped[str] = mapped_column(String)  # training | live | replay
    controller: Mapped[str] = mapped_column(String)  # dqn | webster | max_pressure | actuated
    config: Mapped[dict] = mapped_column(JSON)  # hyperparams, sumocfg ref, ablation flags
    data_version: Mapped[str | None] = mapped_column(String, nullable=True)
    lstm_version: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)  # UUID (provenance)
    sumo_version: Mapped[str | None] = mapped_column(String, nullable=True)
    git_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    episodes: Mapped[list["Episode"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["ModelArtifact"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Episode(Base):
    """One episode within a run, plus the B3 gridlock-guard counters.

    The gridlock-guard columns (open-items B3/F) let eval flag and censor
    episodes where SUMO could not insert the demanded traffic (insertion
    backlog) so a gridlocked run is not silently scored as high-throughput.
    """

    __tablename__ = "episode"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    run_id_fk: Mapped[int] = mapped_column(ForeignKey("experiment_run.id"), index=True)
    index_in_run: Mapped[int] = mapped_column(Integer)  # 0-based episode number
    seed: Mapped[int] = mapped_column(Integer)
    scenario: Mapped[str] = mapped_column(String)  # SCN-01..05
    total_reward: Mapped[float] = mapped_column(Float, default=0.0)
    sim_duration: Mapped[float | None] = mapped_column(Float, nullable=True)  # s
    done_reason: Mapped[str | None] = mapped_column(String, nullable=True)  # time_limit | no_vehicles
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # --- B3 gridlock-guard columns (T-01-03 DoD; open-items B3/F) ---
    loaded_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    departed_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    arrived_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    insertion_backlog_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    gridlock_censored: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped["ExperimentRun"] = relationship(back_populates="episodes")
    observations: Mapped[list["Observation"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )
    kpi: Mapped["EpisodeKpi | None"] = relationship(
        back_populates="episode", cascade="all, delete-orphan", uselist=False
    )
    vehicle_snapshots: Mapped[list["VehicleSnapshot"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )


class Observation(Base):
    """One decision step (every 10 simulated seconds).

    High-volume table (open-items C4): if row counts bite, per-step ``state`` can
    be demoted to JSONL keeping only the KPI columns here. Indexed on
    ``(episode_id_fk, step)``.
    """

    __tablename__ = "observation"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    episode_id_fk: Mapped[int] = mapped_column(
        ForeignKey("episode.id"), index=True
    )
    step: Mapped[int] = mapped_column(Integer)  # decision-step index
    sim_time: Mapped[float] = mapped_column(Float)  # SUMO seconds
    state: Mapped[dict] = mapped_column(JSON)  # 56-d hybrid or 20-d base obs
    action: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0-7; null for baselines
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_abs: Mapped[float | None] = mapped_column(Float, nullable=True)  # |P(s')|
    mask: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # bool[8]

    episode: Mapped["Episode"] = relationship(back_populates="observations")


class EpisodeKpi(Base):
    """Per-episode computed KPIs (data-schema.md s3; definitions locked in kpis.md).

    One row per episode (1:1). Populated by the KPI extractor in Phase 2; the
    table shape is fixed here so eval queries are stable.
    """

    __tablename__ = "episode_kpi"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    episode_id_fk: Mapped[int] = mapped_column(
        ForeignKey("episode.id"), index=True, unique=True
    )
    avg_waiting_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # s/veh
    avg_queue_length: Mapped[float | None] = mapped_column(Float, nullable=True)  # vehicles
    throughput: Mapped[float | None] = mapped_column(Float, nullable=True)  # veh/h
    num_stops: Mapped[float | None] = mapped_column(Float, nullable=True)  # per-veh stop count
    wait_p95: Mapped[float | None] = mapped_column(Float, nullable=True)  # s
    fairness_std: Mapped[float | None] = mapped_column(Float, nullable=True)  # s, stddev across 12 mvts
    per_movement_max_wait: Mapped[list | None] = mapped_column(JSON, nullable=True)  # s[12]
    # E5 (open-items): the per-movement p95 wait alongside the absolute max, and the scalar
    # worst-movement max the eval table/plots query directly (kpi_extractor emits both).
    per_movement_p95_wait: Mapped[list | None] = mapped_column(JSON, nullable=True)  # s[12]
    worst_movement_max_wait: Mapped[float | None] = mapped_column(Float, nullable=True)  # s

    episode: Mapped["Episode"] = relationship(back_populates="kpi")


class ModelArtifact(Base):
    """A saved checkpoint (DQN or LSTM); the filename embeds the version chain (s6)."""

    __tablename__ = "model_artifact"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    run_id_fk: Mapped[int] = mapped_column(ForeignKey("experiment_run.id"), index=True)
    path: Mapped[str] = mapped_column(String)  # checkpoint file on disk
    kind: Mapped[str] = mapped_column(String)  # dqn | lstm
    step: Mapped[int] = mapped_column(Integer)  # training step at save
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["ExperimentRun"] = relationship(back_populates="artifacts")


class VehicleSnapshot(Base):
    """Index row pointing at a JSONL trace - bulk per-vehicle data is NOT here."""

    __tablename__ = "vehicle_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default=SCHEMA_VERSION)
    episode_id_fk: Mapped[int] = mapped_column(
        ForeignKey("episode.id"), index=True
    )
    jsonl_path: Mapped[str] = mapped_column(String)  # pointer to the JSONL trace file
    sim_time: Mapped[float] = mapped_column(Float)  # snapshot time
    vehicle_count: Mapped[int] = mapped_column(Integer)

    episode: Mapped["Episode"] = relationship(back_populates="vehicle_snapshots")
