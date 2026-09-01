"""T-01-01 - Load + validate the SCN-01..05 scenario configuration files.

A *scenario* is a declarative description of traffic demand at the intersection
(``config/scenarios/scn_*.yaml``), per ``notes/03-simulation.md`` §4.5. This
module turns one YAML file into a validated, frozen ``Scenario`` and exposes the
instantaneous arrival rate per axis via ``AxisDemand.rate_at(t)``. The route
generator (T-01-08) consumes these to emit deterministic ``.rou.xml`` files;
this module does NOT itself talk to SUMO.

Demand is given per axis pair (N/S, E/W). Three profile shapes are supported:

* ``constant``     - ``vph`` flat for the whole episode.
* ``ramp``         - linear ``vph_start`` -> ``vph_end`` over ``ramp_s``, then hold.
* ``sinusoidal``   - oscillate ``vph_min``..``vph_max`` with ``period_s`` and
  ``phase_offset_deg`` (lets two axes peak out of phase).

Any malformed file raises ``ScenarioError`` (fail loud at load, never silently
half-load a bad config - notes/03-simulation.md §6 error table).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = _REPO_ROOT / "config" / "scenarios"

_PROFILE_PARAMS: dict[str, tuple[str, ...]] = {
    "constant": ("vph",),
    "ramp": ("vph_start", "vph_end", "ramp_s"),
    "sinusoidal": ("vph_min", "vph_max", "period_s", "phase_offset_deg"),
    # Measured demand. Carries no scalar parameters: its series is loaded from the demand_count
    # table by `source` + `approaches`, so the numbers in a run trace to an ingested dataset
    # rather than to a curve someone chose.
    "tabular": (),
}
_TURNS = frozenset({"left", "through", "right"})


class ScenarioError(ValueError):
    """Raised when a scenario file is missing or malformed."""


@dataclass(frozen=True)
class AxisDemand:
    """Arrival-rate profile for one axis pair (N/S or E/W).

    ``bins`` is populated only for the ``tabular`` profile, where the rate comes from MEASURED
    counts rather than a formula - see :func:`load_measured_bins`.
    """

    profile: str
    params: dict[str, float]
    #: ``tabular`` only: which ingested dataset and which approaches make up this axis. The series
    #: itself is fetched on FIRST USE, not at load time - ``load_all()`` globs every scenario file,
    #: so an eager read would make merely listing the scenarios require a database, and a fresh
    #: clone (which has none) could not even enumerate the synthetic ones.
    source: str | None = None
    approaches: tuple[str, ...] = ()

    @property
    def bins(self) -> tuple[tuple[float, float, float], ...]:
        """Measured ``(start_s, end_s, veh_per_hour)`` rows, cached across calls."""
        if self.profile != "tabular":
            return ()
        return _measured_bins(self.source or "", self.approaches)

    def rate_at(self, t: float) -> float:
        """Instantaneous arrival rate (veh/h) at sim-time ``t`` seconds."""
        if self.profile == "tabular":
            bins = self.bins
            if not bins:
                raise ScenarioError(
                    f"no measured demand for source {self.source!r} approaches "
                    f"{list(self.approaches)}. Ingest it: "
                    f"python -m scripts.ingest_demand --file <flow.json>"
                )
            # Step, not interpolate. Each row is a COUNT over a closed interval - "37 vehicles
            # between 300 s and 600 s" - so the rate is constant across the bin by construction.
            # Interpolating would invent a within-bin shape the measurement does not contain.
            for start, end, rate in bins:
                if start <= t < end:
                    return rate
            # Past the end of the record: hold the last observed rate rather than drop to zero,
            # so an episode longer than the data degrades into "more of the same" instead of
            # into an empty road that would look like a simulator fault.
            return bins[-1][2]
        p = self.params
        if self.profile == "constant":
            return p["vph"]
        if self.profile == "ramp":
            if t >= p["ramp_s"]:
                return p["vph_end"]
            frac = t / p["ramp_s"] if p["ramp_s"] > 0 else 1.0
            return p["vph_start"] + (p["vph_end"] - p["vph_start"]) * frac
        if self.profile == "sinusoidal":
            mid = (p["vph_min"] + p["vph_max"]) / 2.0
            amp = (p["vph_max"] - p["vph_min"]) / 2.0
            phase = math.radians(p["phase_offset_deg"])
            return mid + amp * math.sin(2.0 * math.pi * t / p["period_s"] + phase)
        raise ScenarioError(f"unknown profile {self.profile!r}")  # unreachable post-validation


@dataclass(frozen=True)
class Scenario:
    """A validated scenario configuration.

    Two demand schemas are supported (mutually exclusive, decided by the YAML's
    ``demand`` keys):

    * legacy single-intersection: ``ns`` + ``ew`` axis pair (SCN-01..10);
    * arterial corridor: ``through`` (the W1/E2 corridor entries) +
      ``cross_c1``/``cross_c2`` (each junction's N/S cross-street approaches).

    Unused fields of the other schema are ``None``; ``is_arterial`` tells the
    consumers (route generator, Webster planner) which schema applies.
    """

    id: str
    name: str
    description: str
    duration_s: int
    seeds: tuple[int, ...]
    turn_split: dict[str, float]
    vehicle_type: str
    heavy_fraction: float
    ns: AxisDemand | None = None
    ew: AxisDemand | None = None
    through: AxisDemand | None = None
    cross_c1: AxisDemand | None = None
    cross_c2: AxisDemand | None = None

    @property
    def is_arterial(self) -> bool:
        """True when this scenario uses the arterial corridor demand schema."""
        return self.through is not None


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a single scenario YAML file.

    Raises
    ------
    ScenarioError
        If the file is missing or any field is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"scenario file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path.name}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path.name}: top level must be a mapping")
    return _build(raw, where=path.name)


def load_all(directory: str | Path = SCENARIO_DIR) -> list[Scenario]:
    """Load every ``scn_*.yaml`` in ``directory``, sorted by id."""
    files = sorted(Path(directory).glob("scn_*.yaml"))
    if not files:
        raise ScenarioError(f"no scn_*.yaml files in {directory}")
    return sorted((load_scenario(f) for f in files), key=lambda s: s.id)


def _build(raw: dict, where: str) -> Scenario:
    """Validate the raw mapping and construct the frozen ``Scenario``."""

    def require(key: str):
        if key not in raw:
            raise ScenarioError(f"{where}: missing required key {key!r}")
        return raw[key]

    duration_s = require("duration_s")
    if not isinstance(duration_s, int) or duration_s <= 0:
        raise ScenarioError(f"{where}: duration_s must be a positive int, got {duration_s!r}")

    seeds = require("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise ScenarioError(f"{where}: seeds must be a non-empty list of ints")
    if len(set(seeds)) != len(seeds):
        raise ScenarioError(f"{where}: seeds must be unique")

    turn_split = require("turn_split")
    if not isinstance(turn_split, dict) or set(turn_split) != _TURNS:
        raise ScenarioError(f"{where}: turn_split must have keys {sorted(_TURNS)}")
    if any(not _is_number(v) or v < 0 for v in turn_split.values()):
        raise ScenarioError(f"{where}: turn_split values must be non-negative numbers")
    if not math.isclose(sum(turn_split.values()), 1.0, abs_tol=1e-6):
        raise ScenarioError(f"{where}: turn_split must sum to 1.0, got {sum(turn_split.values())}")

    vehicle = require("vehicle")
    if not isinstance(vehicle, dict):
        raise ScenarioError(f"{where}: vehicle must be a mapping")
    heavy_fraction = vehicle.get("heavy_fraction", 0.0)
    if not _is_number(heavy_fraction) or not 0.0 <= heavy_fraction <= 1.0:
        raise ScenarioError(f"{where}: vehicle.heavy_fraction must be in [0, 1]")

    demand = require("demand")
    _LEGACY_KEYS = {"ns", "ew"}
    _ARTERIAL_KEYS = {"through", "cross_c1", "cross_c2"}
    if not isinstance(demand, dict) or set(demand) not in (_LEGACY_KEYS, _ARTERIAL_KEYS):
        raise ScenarioError(
            f"{where}: demand must have exactly keys {sorted(_LEGACY_KEYS)} (legacy) "
            f"or {sorted(_ARTERIAL_KEYS)} (arterial)"
        )

    axes = {key: _build_axis(demand[key], where, key) for key in demand}
    return Scenario(
        id=str(require("id")),
        name=str(require("name")),
        description=str(raw.get("description", "")),
        duration_s=duration_s,
        seeds=tuple(seeds),
        turn_split={k: float(v) for k, v in turn_split.items()},
        vehicle_type=str(vehicle.get("type", "passenger")),
        heavy_fraction=float(heavy_fraction),
        ns=axes.get("ns"),
        ew=axes.get("ew"),
        through=axes.get("through"),
        cross_c1=axes.get("cross_c1"),
        cross_c2=axes.get("cross_c2"),
    )


def _build_axis(raw: object, where: str, axis: str) -> AxisDemand:
    """Validate one axis's demand sub-mapping."""
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: demand.{axis} must be a mapping")
    profile = raw.get("profile")
    if profile not in _PROFILE_PARAMS:
        raise ScenarioError(
            f"{where}: demand.{axis}.profile must be one of {sorted(_PROFILE_PARAMS)}, got {profile!r}"
        )
    params: dict[str, float] = {}
    for key in _PROFILE_PARAMS[profile]:
        if key not in raw:
            raise ScenarioError(f"{where}: demand.{axis} ({profile}) missing {key!r}")
        if not _is_number(raw[key]):
            raise ScenarioError(f"{where}: demand.{axis}.{key} must be a number, got {raw[key]!r}")
        params[key] = float(raw[key])
    # Rates must be non-negative (phase_offset_deg is exempt — it's an angle).
    for key, val in params.items():
        if key != "phase_offset_deg" and val < 0:
            raise ScenarioError(f"{where}: demand.{axis}.{key} must be >= 0, got {val}")
    if profile == "tabular":
        source = raw.get("source")
        approaches = raw.get("approaches")
        if not isinstance(source, str) or not source:
            raise ScenarioError(f"{where}: demand.{axis} (tabular) needs a 'source' dataset key")
        if not isinstance(approaches, list) or not approaches:
            raise ScenarioError(f"{where}: demand.{axis} (tabular) needs a non-empty 'approaches'")
        # Shape is validated here; the data is read on first use (see AxisDemand.bins).
        return AxisDemand(
            profile=profile, params={}, source=source,
            approaches=tuple(str(a) for a in approaches),
        )
    return AxisDemand(profile=profile, params=params)


@lru_cache(maxsize=32)
def _measured_bins(source: str, approaches: tuple[str, ...]):
    """Cached wrapper - the route generator asks for a rate thousands of times per episode."""
    return load_measured_bins(source, list(approaches))


def load_measured_bins(
    source: str, approaches: list[str], db_path: Path | None = None
) -> tuple[tuple[float, float, float], ...]:
    """Measured arrival rate per time bin for one axis, read from the results database.

    The per-approach counts are **averaged**, not summed, and that is not a detail:
    ``scripts/build_routes.py`` gives EVERY approach on an axis the full axis rate
    (``_APPROACH_AXIS``), so summing N and S here would double the traffic the dataset actually
    recorded.

    Imported lazily so that loading a scenario keeps working with no database present - only the
    ``tabular`` profile needs one, and the other four profiles are pure arithmetic.
    """
    from sqlalchemy import func as _func, select
    from sqlalchemy.orm import Session

    from src.db.engine import create_db_engine
    from src.db.models import DemandCount

    path = db_path or (_REPO_ROOT / "data" / "traffic.db")
    if not Path(path).exists():
        raise ScenarioError(
            f"tabular demand needs the results database at {path}, which does not exist. "
            f"Run: python -m scripts.ingest_demand --file <flow.json>"
        )

    engine = create_db_engine(Path(path))
    with Session(engine) as session:
        rows = session.execute(
            select(
                DemandCount.bin_start_s,
                DemandCount.bin_end_s,
                _func.avg(DemandCount.vehicles),
            )
            .where(DemandCount.source == source, DemandCount.approach.in_(approaches))
            .group_by(DemandCount.bin_start_s, DemandCount.bin_end_s)
            .order_by(DemandCount.bin_start_s)
        ).all()

    out = []
    for start, end, mean_vehicles in rows:
        span = float(end) - float(start)
        rate = 0.0 if span <= 0 else float(mean_vehicles) * 3600.0 / span
        out.append((float(start), float(end), rate))
    return tuple(out)


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
