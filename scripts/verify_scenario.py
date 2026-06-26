"""Forecaster-rescue #0 - Verify candidate test-scenario feasibility for the TRAINED DQN too.

The baseline-only calibration is insufficient: the eval showed the DQN gridlocks far more than
the baselines (SCN-05: DQN 93% vs Webster 20%). A test scenario with statistical power needs the
DQN itself to mostly NOT gridlock. This probes candidate sinusoidal demands with Webster +
actuated + a trained plain DQN + a trained hybrid DQN (seed 123, the non-gridlock-prone seed; s42
was the fragile one), reporting the per-controller gridlock-censoring rate over 6 seeds. Pick the
HIGHEST-demand candidate where the DQN censoring is acceptable (~<=30%) so n stays usable.

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.verify_scenario
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_runner import _OFFICIAL_LSTM, Algo, _load_agent, run_eval_episode
from src.baselines.max_pressure import MaxPressureController  # noqa: F401 (kept for parity)
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.ml.hybrid_wrapper import load_forecaster
from src.scenarios.config import AxisDemand, Scenario

_OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "calib"
_RUNS = Path(__file__).resolve().parent.parent / "runs"
SEEDS = [0, 1, 2, 3, 4, 5]
TURN = {"left": 0.2, "through": 0.6, "right": 0.2}

# (label, vph_min, vph_max) sinusoidal both axes, 90deg offset.
CANDIDATES = [
    ("m1000:150-350", 150, 350),
    ("m1100:150-400", 150, 400),
    ("m1200:150-450", 150, 450),
]


def _scn(vmin: int, vmax: int) -> Scenario:
    def axis(off: float) -> AxisDemand:
        return AxisDemand("sinusoidal", {"vph_min": float(vmin), "vph_max": float(vmax),
                                         "period_s": 3600.0, "phase_offset_deg": off})
    return Scenario(id=f"SCN-V{vmin}{vmax}", name="verify", description="", duration_s=3600,
                    seeds=tuple(SEEDS), turn_split=TURN, vehicle_type="passenger",
                    heavy_fraction=0.0, ns=axis(0.0), ew=axis(90.0))


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = _load_agent(_RUNS / "plain_seed123" / "checkpoints" / "ep299.pt", 20)
    hybrid = _load_agent(_RUNS / "hybrid_seed123" / "checkpoints" / "ep299.pt", 56)
    forecaster = load_forecaster(str(_OFFICIAL_LSTM))

    print(f"{'candidate':16}  webster  actuated  dqn-plain  dqn-hybrid   (gridlock-censor % over 6 seeds)")
    print("-" * 86)
    for label, vmin, vmax in CANDIDATES:
        scn = _scn(vmin, vmax)
        algos = [
            Algo("webster", "baseline", controller=WebsterController(webster_plan_for_scenario(scn))),
            Algo("actuated", "baseline", signal_mode="actuated"),
            Algo("dqn-plain", "dqn", agent=plain),
            Algo("dqn-hybrid", "dqn", agent=hybrid, forecaster=forecaster),
        ]
        cells = []
        for algo in algos:
            cens = []
            for s in SEEDS:
                kpis, _r = run_eval_episode(scn, s, algo, work_dir=_OUT,
                                            episode_length_s=3600, warmup_s=300.0)
                cens.append(int(kpis.gridlock_censored))
            cells.append(f"{100 * np.mean(cens):5.0f}%")
        print(f"{label:16}  {cells[0]:>7}  {cells[1]:>8}  {cells[2]:>9}  {cells[3]:>10}")
    print("\nPick the highest demand with DQN censoring <=~30% on BOTH plain & hybrid.")


if __name__ == "__main__":
    main()
