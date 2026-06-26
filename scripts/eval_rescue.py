"""Forecaster-rescue #1 (final) - Does the DQN USE a strong forecast? plain vs hybrid vs hybrid-boot.

The decisive test of the whole rescue. On the powered SCN-06 (n=15, 0% gridlock), compares:
  * plain            - 20-dim DQN, no forecast (the control)
  * hybrid-official  - 56-dim DQN + the Webster-trained forecaster (skill ~0 on SCN-06)
  * hybrid-boot      - 56-dim DQN + the DQN-bootstrapped forecaster (skill +0.45..0.57 on SCN-06)

All evaluated GREEDY from their final ep299 checkpoints on the SAME held-out eval seeds (paired).
Reports per-variant mean wait/throughput + the two-sided Wilcoxon signed-rank (Pratt) for the
headline pairings hybrid-boot-vs-plain and hybrid-boot-vs-hybrid-official, paired by
(train_seed, eval_seed) -> n=15. If hybrid-boot beats plain, a forecaster WITH skill on the test
regime helps the controller (rescue succeeds, with the caveat it must be trained on representative
regimes). If not, the DQN cannot exploit even a strong forecast (a strong negative finding).

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.eval_rescue
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_eval import _wilcoxon
from scripts.build_network import build_net
from scripts.eval_runner import _OFFICIAL_LSTM, Algo, _load_agent, run_eval_episode
from src.ml.hybrid_wrapper import load_forecaster
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT = _REPO_ROOT / "data" / "eval" / "rescue"
_BOOT_LSTM = _REPO_ROOT / "checkpoints" / "lstm" / "lstm-dqn-bootstrap.pt"
SEEDS = (42, 123, 2024)
EVAL_SEEDS = (7000, 7001, 7002, 7003, 7004)

# variant -> (run-dir prefix, obs_dim, forecaster-kind)
VARIANTS = {
    "plain": ("plain", 20, None),
    "hybrid-official": ("hybrid", 56, "official"),
    "hybrid-boot": ("hybrid-boot", 56, "boot"),
}


def _forecaster(kind):
    if kind == "official":
        return load_forecaster(str(_OFFICIAL_LSTM))
    if kind == "boot":
        return load_forecaster(str(_BOOT_LSTM))
    return None


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    scn = load_scenario(SCENARIO_DIR / "scn_06.yaml")

    # results[variant][(train_seed, eval_seed)] = (avg_wait, throughput, censored)
    results: dict[str, dict] = {v: {} for v in VARIANTS}
    for variant, (prefix, obs_dim, fckind) in VARIANTS.items():
        forecaster = _forecaster(fckind)
        for ts in SEEDS:
            ckpt = _RUNS / f"{prefix}_seed{ts}" / "checkpoints" / "ep299.pt"
            if not ckpt.exists():
                print(f"[rescue] MISSING {ckpt} - run the hybrid-boot retrain first.")
                return
            agent = _load_agent(ckpt, obs_dim)
            algo = Algo(f"{variant}-s{ts}", "dqn", agent=agent, forecaster=forecaster)
            for es in EVAL_SEEDS:
                kpis, _r = run_eval_episode(scn, es, algo, work_dir=_OUT,
                                            episode_length_s=3600, warmup_s=300.0)
                results[variant][(ts, es)] = (kpis.avg_waiting_time, kpis.throughput,
                                              int(kpis.gridlock_censored))
            print(f"[rescue] {variant:16} seed {ts} done")

    keys = [(ts, es) for ts in SEEDS for es in EVAL_SEEDS]
    print("\n=== SCN-06 (powered, n=15) — mean over valid (non-censored) episodes ===")
    print(f"{'variant':16}  avg_wait(s)   throughput   %censored")
    for v in VARIANTS:
        valid = [results[v][k] for k in keys if results[v][k][2] == 0]
        waits = [w for w, _t, _c in valid if not np.isnan(w)]
        thru = [t for _w, t, _c in valid if not np.isnan(t)]
        cens = 100 * np.mean([results[v][k][2] for k in keys])
        print(f"{v:16}  {np.mean(waits):9.2f}   {np.mean(thru):9.1f}   {cens:6.0f}%")

    def _paired(a_var, b_var, idx):
        a, b = [], []
        for k in keys:
            va, vb = results[a_var][k][idx], results[b_var][k][idx]
            if not (np.isnan(va) or np.isnan(vb)):
                a.append(va)
                b.append(vb)
        return np.array(a), np.array(b)

    print("\n=== DECISIVE Wilcoxon (two-sided, paired n<=15) ===")
    for a_var, b_var in [("hybrid-boot", "plain"), ("hybrid-boot", "hybrid-official")]:
        for idx, label, better in [(1, "throughput", "higher"), (0, "avg_wait", "lower")]:
            a, b = _paired(a_var, b_var, idx)
            p, med, lo, hi, n = _wilcoxon(a, b)
            sig = "SIGNIFICANT" if (not np.isnan(p) and p < 0.05) else "n.s."
            print(f"  {a_var} vs {b_var:16} {label:11} ({better}): "
                  f"median d={med:+.1f} [{lo:+.1f},{hi:+.1f}] n={n} p={p:.4f}  {sig}")


if __name__ == "__main__":
    main()
