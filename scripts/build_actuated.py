"""T-02-06 - Generate the actuated baseline's additional-file (program + detectors).

SUMO's ``actuated`` traffic-light type runs gap-based control entirely inside
SUMO (baselines-implementation.md Baseline 3): it holds a green while vehicles
keep crossing a detector (gap < ``max-gap``) up to ``maxDur``, and switches at
``minDur`` once the gap opens. Unlike the RL env, Python never commands the lights.

This script writes ``config/network/actuated.add.xml`` deterministically:

* an ``actuated`` ``tlLogic`` for ``C`` cycling the **same 8 NEMA green phases** as
  the RL env (reusing :class:`Intersection` green/yellow/all-red synthesis, so the
  phase set is identical and cannot drift), with ``minDur=10`` / ``maxDur=60`` and
  ``max-gap=3.0`` pinned identically (research-eval-repro Round 2 / open-items);
* explicit induction-loop detectors, one per controlled incoming lane, at a pinned
  position - "declare detectors explicitly" for reproducibility (research-sumo §4).

Custom detectors only take effect when the ``tlLogic`` is loaded from an additional
file (SUMO docs), which is exactly how :class:`~src.env.sumo_env.SUMOEnv` loads it
in actuated mode (``-a`` + ``setProgram``).

Run::

    python -m scripts.build_actuated
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import traci
from sumolib import checkBinary

from src.env.intersection import _VAULT_MOVEMENTS, Intersection

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NET_DIR = _REPO_ROOT / "config" / "network"
_NET_FILE = _NET_DIR / "intersection.net.xml"
_ADD_FILE = _NET_DIR / "actuated.add.xml"

TLS_ID = "C"
PROGRAM_ID = "actuated"
CYCLE = (0, 1, 2, 3, 4, 5, 6, 7)  # serve the 8 NEMA phases in NS-then-EW order

MIN_GREEN_S = 10
MAX_GREEN_S = 60
MAX_GAP_S = 3.0  # gap-out threshold (pinned across scenarios)
DETECTOR_POS = -20.0  # m from the lane end (stop line); pinned, deterministic
# SUMO requires a `file` attribute; the gap logic ignores its output. NUL discards
# it on Windows (this project's pinned platform); override for other OSes.
DETECTOR_OUTPUT = "NUL"


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print helper (stdlib has no indent before 3.9 API on Element)."""
    pad = "\n" + "    " * level
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = pad + "    "
        for child in elem:
            _indent(child, level + 1)
            if not (child.tail and child.tail.strip()):
                child.tail = pad + "    "
        if not (elem[-1].tail and elem[-1].tail.strip()):
            elem[-1].tail = pad
    if level and not (elem.tail and elem.tail.strip()):
        elem.tail = pad


def build_actuated_add(
    *,
    out_path: Path = _ADD_FILE,
    net_file: Path = _NET_FILE,
    movements_path: Path = _VAULT_MOVEMENTS,
) -> Path:
    """Build and write the actuated additional-file; return its path."""
    traci.start([checkBinary("sumo"), "-n", str(net_file), "--no-step-log", "true"])
    try:
        ix = Intersection.from_traci(traci, TLS_ID, movements_path=movements_path)
        lanes = ix.controlled_in_lanes

        root = ET.Element("additional")

        # one induction-loop detector per controlled incoming lane
        detector_for: dict[str, str] = {}
        for lane in lanes:
            det_id = f"det_{lane}"
            detector_for[lane] = det_id
            ET.SubElement(
                root,
                "inductionLoop",
                id=det_id,
                lane=lane,
                pos=f"{DETECTOR_POS}",
                period="100000",  # effectively no periodic output
                file=DETECTOR_OUTPUT,
            )

        tl = ET.SubElement(
            root, "tlLogic", id=TLS_ID, programID=PROGRAM_ID, offset="0", type="actuated"
        )
        ET.SubElement(tl, "param", key="max-gap", value=f"{MAX_GAP_S}")
        # bind each controlled lane to its explicit detector (key = lane id)
        for lane in lanes:
            ET.SubElement(tl, "param", key=lane, value=detector_for[lane])

        # phase sequence: green(minDur/maxDur) -> yellow(3s) -> [all-red 2s on barrier]
        for i, action in enumerate(CYCLE):
            nxt = CYCLE[(i + 1) % len(CYCLE)]
            ET.SubElement(
                tl, "phase",
                duration=f"{MIN_GREEN_S}", minDur=f"{MIN_GREEN_S}", maxDur=f"{MAX_GREEN_S}",
                state=ix.green_state(action),
            )
            ET.SubElement(
                tl, "phase",
                duration=f"{ix.yellow_s}", minDur=f"{ix.yellow_s}", maxDur=f"{ix.yellow_s}",
                state=ix.yellow_state(action, nxt),
            )
            if ix.is_barrier_crossing(action, nxt):
                ET.SubElement(
                    tl, "phase",
                    duration=f"{ix.all_red_s}", minDur=f"{ix.all_red_s}", maxDur=f"{ix.all_red_s}",
                    state=ix.all_red_state(),
                )
    finally:
        traci.close()

    _indent(root)
    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- Generated by scripts/build_actuated.py (T-02-06) - do not edit by hand. -->\n"
        + ET.tostring(root, encoding="unicode")
        + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movements", type=Path, default=_VAULT_MOVEMENTS)
    args = parser.parse_args()
    out = build_actuated_add(movements_path=args.movements)
    print(f"[actuated] wrote {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
