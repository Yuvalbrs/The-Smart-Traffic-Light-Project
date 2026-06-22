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
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = _REPO_ROOT / "config" / "scenarios"

_PROFILE_PARAMS: dict[str, tuple[str, ...]] = {
    "constant": ("vph",),
    "ramp": ("vph_start", "vph_end", "ramp_s"),
    "sinusoidal": ("vph_min", "vph_max", "period_s", "phase_offset_deg"),
}
_TURNS = frozenset({"left", "through", "right"})


class ScenarioError(ValueError):
    """Raised when a scenario file is missing or malformed."""


@dataclass(frozen=True)
class AxisDemand:
    """Arrival-rate profile for one axis pair (N/S or E/W)."""

    profile: str
    params: dict[str, float]

    def rate_at(self, t: float) -> float:
        """Instantaneous arrival rate (veh/h) at sim-time ``t`` seconds."""
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
    """A validated scenario configuration."""

    id: str
    name: str
    description: str
    duration_s: int
    seeds: tuple[int, ...]
    turn_split: dict[str, float]
    vehicle_type: str
    heavy_fraction: float
    ns: AxisDemand
    ew: AxisDemand


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
    if not isinstance(demand, dict) or set(demand) != {"ns", "ew"}:
        raise ScenarioError(f"{where}: demand must have exactly keys 'ns' and 'ew'")

    return Scenario(
        id=str(require("id")),
        name=str(require("name")),
        description=str(raw.get("description", "")),
        duration_s=duration_s,
        seeds=tuple(seeds),
        turn_split={k: float(v) for k, v in turn_split.items()},
        vehicle_type=str(vehicle.get("type", "passenger")),
        heavy_fraction=float(heavy_fraction),
        ns=_build_axis(demand["ns"], where, "ns"),
        ew=_build_axis(demand["ew"], where, "ew"),
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
    return AxisDemand(profile=profile, params=params)


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
