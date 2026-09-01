"""Ingest REAL measured traffic counts into the results database.

Everything this project has simulated so far was driven by a formula: a scenario YAML says
"sinusoidal, 200-600 vph, 90 degrees out of phase" and the route generator samples arrivals from
it. That is reproducible and controllable, and it is not measured. This script closes that gap -
it reads a published dataset of real vehicle arrivals, normalises it onto this project's four
approaches, and writes it into the ``demand_count`` table, from which a scenario's traffic can be
generated instead of from a curve.

**Source: the Hangzhou 1x1 benchmark** (`flow.json`), the demand set used by PressLight, MPLight,
CityFlow and LibSignal. Chosen over larger municipal datasets because this project is explicitly a
replication-plus-adaptation of MPLight, so its own reference papers already calibrate against this
data - using anything else would need defending as a deviation.

Two properties of the file that are easy to get wrong, both verified against the bytes:

* every entry has ``endTime == startTime``, so each object is exactly ONE vehicle. The ``interval``
  field (always 5) is CityFlow's "repeat every N seconds" knob and is inert here; multiplying by it
  inflates demand five-fold.
* ``route`` is always ``[entry_road, exit_road]`` and the entry road names the approach.

Road ids are ``road_<x>_<y>_<dir>`` where dir is 0=E, 1=N, 2=W, 3=S and names the direction the
road *leaves* its node in. The four entry roads therefore feed the intersection from the opposite
compass point: ``road_0_1_0`` runs east out of the west node, i.e. it is the WEST approach.

**Licence, stated because it is a finding rather than a footnote:** neither the LibSignal nor the
sample-code repository carries a LICENSE file. The benchmark's own page asks only that users cite
Wei et al. (2019) and Zheng et al. (2019). That is the operative term - a citation request, not an
open-licence grant - and it is recorded on every row.

Run::

    python -m scripts.ingest_demand --file data/external/hangzhou_bc-tyc_18041607_flow.json
    python -m scripts.ingest_demand --file <path> --source my_counts --bin-seconds 300
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from sqlalchemy.orm import Session

from src.db.engine import create_db_engine, init_db
from src.db.models import DemandCount

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "traffic.db"

SOURCE_URL = (
    "https://raw.githubusercontent.com/DaRL-LibSignal/LibSignal/master/data/raw_data/"
    "hangzhou_1x1_bc-tyc_18041607_1h/flow.json"
)
SOURCE_TERMS = (
    "No LICENSE file in the source repository. The benchmark asks that users cite "
    "Wei et al. 2019 (PressLight) and Zheng et al. 2019 (CityFlow). Citation request, "
    "not an open-licence grant."
)

#: Last digit of a CityFlow road id -> the direction it points. A road pointing east is entered
#: from the west, so the approach is the OPPOSITE compass point.
_DIRECTION = {"0": "E", "1": "N", "2": "W", "3": "S"}
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

APPROACHES = ("N", "E", "S", "W")


def approach_of(road_id: str) -> str:
    """``"road_0_1_0"`` -> ``"W"``: the approach a vehicle entering on that road arrives from."""
    direction = _DIRECTION.get(road_id.rsplit("_", 1)[-1])
    if direction is None:
        raise ValueError(f"cannot read a direction out of road id {road_id!r}")
    return _OPPOSITE[direction]


def read_flow(path: Path) -> list[tuple[float, str]]:
    """``flow.json`` -> ``[(depart_seconds, approach), ...]``, one tuple per vehicle."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"{path} is not a CityFlow flow.json (expected a JSON array)")

    out: list[tuple[float, str]] = []
    for i, entry in enumerate(entries):
        route = entry.get("route") or []
        if not route:
            raise SystemExit(f"{path}: entry {i} has no route")
        start, end = float(entry["startTime"]), float(entry["endTime"])
        if end != start:
            # A span would mean "emit repeatedly every `interval` seconds". Every entry in the
            # shipped benchmark is a single vehicle; refuse rather than silently under-count.
            raise SystemExit(
                f"{path}: entry {i} spans {start}-{end}s. This reader assumes one vehicle per "
                f"entry (endTime == startTime), which holds for the Hangzhou benchmark files."
            )
        out.append((start, approach_of(route[0])))
    return out


def bin_counts(
    vehicles: Iterable[tuple[float, str]], bin_seconds: float
) -> list[tuple[str, float, float, int]]:
    """Bin ``(depart, approach)`` pairs into ``(approach, start, end, count)`` rows.

    Every bin of every approach is emitted, including empty ones: a missing row and a zero are
    different claims, and a demand profile that silently skips a quiet bin would interpolate
    across it as though traffic never stopped.
    """
    vehicles = list(vehicles)
    if not vehicles:
        return []
    horizon = max(t for t, _ in vehicles)
    n_bins = int(horizon // bin_seconds) + 1

    counts: Counter[tuple[str, int]] = Counter()
    for depart, approach in vehicles:
        counts[(approach, int(depart // bin_seconds))] += 1

    rows = []
    for approach in APPROACHES:
        for b in range(n_bins):
            rows.append((
                approach, b * bin_seconds, (b + 1) * bin_seconds, counts[(approach, b)],
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file", type=Path, required=True, help="CityFlow flow.json")
    parser.add_argument("--source", default=None, help="dataset key (default: the file stem)")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--bin-seconds", type=float, default=300.0,
                        help="aggregation interval; 300 s matches how detector data is published")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--replace", action="store_true",
                        help="delete existing rows for this source first (re-ingest)")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"{args.file} not found. Download it first:\n  curl -L -o {args.file} {SOURCE_URL}")
    source = args.source or args.file.stem.replace("_flow", "")

    vehicles = read_flow(args.file)
    rows = bin_counts(vehicles, args.bin_seconds)

    engine = create_db_engine(args.db)
    init_db(engine)
    with Session(engine) as session:
        existing = session.query(DemandCount).filter(DemandCount.source == source).count()
        if existing and not args.replace:
            raise SystemExit(
                f"{existing} rows already ingested for source {source!r}. Pass --replace to "
                f"overwrite, or --source to ingest under a different key."
            )
        if existing:
            session.query(DemandCount).filter(DemandCount.source == source).delete()

        session.add_all([
            DemandCount(
                source=source, source_url=args.source_url, source_terms=SOURCE_TERMS,
                approach=approach, bin_start_s=start, bin_end_s=end, vehicles=n,
            )
            for approach, start, end, n in rows
        ])
        session.commit()

    per_approach = Counter(a for _, a in vehicles)
    horizon = max(t for t, _ in vehicles)
    print(f"[ingest] {args.file.name}: {len(vehicles)} vehicles over {horizon:.0f}s")
    print(f"[ingest] per approach: " + "  ".join(f"{a}={per_approach[a]}" for a in APPROACHES))
    print(f"[ingest] wrote {len(rows)} bins ({args.bin_seconds:.0f}s each) as source {source!r}")
    print(f"[ingest] -> {args.db}")


if __name__ == "__main__":
    main()
