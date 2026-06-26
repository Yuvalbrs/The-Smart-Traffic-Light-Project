"""Forecaster-rescue #0 - Calibrate a near-saturation, still-SHIFTING test scenario.

The original SCN-05 (sinusoidal 200-600 vph/axis, 90deg offset) peaks at ~2166 veh/h total -
far above the ~1650 veh/h the intersection can clear (SCN-04 evidence) - so 78% of episodes
gridlock and the eval has no statistical power. This sweep finds a sinusoidal demand whose PEAK
load sits just below saturation (target ~85-90%), keeping the out-of-phase oscillation (where a
60-120s forecast can actually pay off) but letting controllers stay un-gridlocked so a real
hybrid-vs-plain effect is detectable.

Probe = the two most robust baselines (Webster, SUMO-actuated). A candidate is "feasible" when
BOTH gridlock-censor on <~10-15% of episodes (margin for the gridlock-prone DQN). Pick the
HIGHEST-demand feasible candidate. Prints a table; writes nothing (the chosen demand is hand-set
into config/scenarios/scn_06.yaml after).

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.calibrate_scenario
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_baselines import run_episode
from src.scenarios.config import AxisDemand, Scenario

_OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "calib"
SEEDS = [0, 1, 2, 3, 4, 5]  # 6 seeds for a stable censoring estimate
TURN = {"left": 0.2, "through": 0.6, "right": 0.2}

# (label, vph_min, vph_max) - sinusoidal both axes, 90deg out of phase (like SCN-05).
# Isolating mean (4*mid) vs swing (amp) near the feasible frontier; we want MAX swing that
# still stays un-gridlocked (swing = where a 60-120s forecast pays off).
CANDIDATES = [
    ("d:250-450 m1400 a100", 250, 450),  # high mean, low swing - the clean 3-seed candidate
    ("e:200-450 m1300 a125", 200, 450),
    ("i:225-475 m1400 a125", 225, 475),  # mean 1400, more swing than d
    ("h:200-500 m1400 a150", 200, 500),  # mean 1400, max swing (peak ~1980 - likely censors)
    ("f:175-425 m1200 a125", 175, 425),  # lower mean, good swing - safety pick
]


def _make(vmin: int, vmax: int, cid: str) -> Scenario:
    def axis(offset: float) -> AxisDemand:
        return AxisDemand("sinusoidal", {
            "vph_min": float(vmin), "vph_max": float(vmax),
            "period_s": 3600.0, "phase_offset_deg": offset,
        })
    return Scenario(
        id=cid, name="calib", description="", duration_s=3600, seeds=tuple(SEEDS),
        turn_split=TURN, vehicle_type="passenger", heavy_fraction=0.0,
        ns=axis(0.0), ew=axis(90.0),
    )


def _peak_total(vmin: int, vmax: int) -> float:
    """Max instantaneous total demand (veh/h) across both axes (both approaches per axis)."""
    mid, amp = (vmin + vmax) / 2, (vmax - vmin) / 2
    return 4 * mid + 2 * amp * np.sqrt(2)  # 2 approaches/axis; axes 90deg out of phase


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()
    print(f"{'candidate':16}{'peak_total':>11}   webster cens%/thru    actuated cens%/thru")
    print("-" * 70)
    for label, vmin, vmax in CANDIDATES:
        scn = _make(vmin, vmax, f"SCN-CAL{vmin}{vmax}")
        cells = []
        for ctrl in ("webster", "actuated"):
            cens, thru = [], []
            for s in SEEDS:
                k = run_episode(scn, s, ctrl, work_dir=_OUT, episode_length_s=3600, warmup_s=300.0)
                cens.append(int(k.gridlock_censored))
                thru.append(k.throughput)
            cells.append(f"{100 * np.mean(cens):4.0f}% / {np.nanmean(thru):6.0f}")
        print(f"{label:16}{_peak_total(vmin, vmax):11.0f}   {cells[0]:>16}    {cells[1]:>16}")
    print("\nPick the HIGHEST-demand row where BOTH baselines censor <~15%.")


if __name__ == "__main__":
    main()
