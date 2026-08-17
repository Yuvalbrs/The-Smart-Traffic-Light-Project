"""Phase-0 diagnostic: is SCN-05 gridlock EXIT-EDGE SPILLBACK or INSERTION BACKLOG?

Question (pre-registered in ``finish-plan.md`` §Phase 0, decided 2026-08-17 BEFORE this
measurement existed). The one structurally new gridlock fix left on the table is
inference-time **exit-occupancy action masking** (OverFlowLight-style): forbid phases that
discharge into an exit edge which is already full. That fix presumes the failure mode is
**exit-edge spillback** — vehicles cannot leave the junction because downstream is jammed.

Sess17 diagnostics instead recorded the failure as **insertion backlog on the approaches**
(demand cannot enter the network at all; the junction starves). If that is what is happening,
an exit mask never binds, and a non-binding constraint is a no-op — a scar this project has
already been struck by twice.

Argument cannot settle this; a measurement can. For each (train seed, eval seed) we drive
SCN-05 with the greedy plain-DQN ep299 policy and log, per decision step:

  * ``traci.edge.getLastStepOccupancy`` for the 4 EXIT edges  (t_n, t_e, t_s, t_w)
  * ``traci.edge.getLastStepOccupancy`` for the 4 APPROACH edges (n_t, e_t, s_t, w_t)
  * the instantaneous insertion backlog = ``len(traci.simulation.getPendingVehicles())``
  * cumulative departures + total standing queue (congestion context)

Episodes are flagged gridlock by ``insertion_backlog_fraction > 0.10`` — the same rule the KPI
extractor censors on, so this diagnostic and the eval agree on what "gridlock" means.

PRE-REGISTERED VERDICT RULE (thresholds below are from ``finish-plan.md`` §Phase 0 and are
fixed in code before the first run; see ``verdict()``):

  * exit occupancy stays **< 0.70** while insertion backlog grows  → candidate **DEAD**. Record
    in ``decisions.md``; the measurement becomes report evidence hardening "gridlock is intrinsic".
  * exit occupancy **>= 0.85** sustained *before* gridlock onset  → ONE timeboxed (<=1 day)
    attempt at an exit-occupancy mask, threshold sweep {0.7, 0.8, 0.9}, per-scenario x per-seed
    eval vs sel/plain.
  * anything in between → **INCONCLUSIVE**, which per §Phase 0 closes the question the same way
    a DEAD verdict does (no further gridlock work either way).

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.diag_exit_occupancy
    LIBSUMO_AS_TRACI=1 python -m scripts.diag_exit_occupancy --seeds 42 --eval-seeds 7000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.env.sumo_env import SUMOEnv
from src.ml.dqn import OBS_DIM, DQNAgent
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT_DIR = _REPO_ROOT / "data" / "eval" / "analysis"
_OUT_CSV = _OUT_DIR / "diag_exit_occupancy.csv"
_OUT_PLOT = _OUT_DIR / "P6_exit_occupancy_vs_backlog.png"
_OUT_MD = _OUT_DIR / "diag_exit_occupancy_verdict.md"

_MASK_FILL = -1e9
_GRIDLOCK_BACKLOG = 0.10  # same threshold the KPI extractor flags gridlock_censored at

EXIT_EDGES = ("t_n", "t_e", "t_s", "t_w")
APPROACH_EDGES = ("n_t", "e_t", "s_t", "w_t")

# --- pre-registered verdict constants (finish-plan.md §Phase 0) ---
DEAD_BELOW = 0.70  # exit occupancy never reaching this while backlog grows => candidate dead
ATTEMPT_AT = 0.85  # exit occupancy sustained at/above this BEFORE onset => one timeboxed attempt
ONSET_PENDING = 5  # >= this many vehicles unable to insert marks gridlock onset...
ONSET_WINDOW = 6  # ...sustained over this many decision steps (6 x 10 s = 60 s)
SUSTAIN_WINDOW = 6  # "sustained" occupancy = mean over a 60 s rolling window


def _load_plain(seed: int) -> DQNAgent:
    """Load the ep299 plain-DQN checkpoint for one training seed, in eval mode."""
    agent = DQNAgent(OBS_DIM)
    state = torch.load(_RUNS / f"plain_seed{seed}" / "checkpoints" / "ep299.pt", map_location="cpu")
    agent.online.load_state_dict(state["online"])
    agent.online.eval()
    return agent


def _q_values(agent: DQNAgent, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        return agent.online(t).squeeze(0).numpy()


def _occupancy(conn: Any, edges: tuple[str, ...]) -> np.ndarray:
    """Occupancy per edge as a FRACTION in [0, 1].

    SUMO has historically reported ``getLastStepOccupancy`` both as a fraction and as a
    percentage depending on version/binding, so normalize defensively rather than trusting
    one convention: any value > 1.0 is treated as a percentage.
    """
    vals = np.array([float(conn.edge.getLastStepOccupancy(e)) for e in edges], dtype=np.float64)
    if np.any(vals > 1.0):
        vals = vals / 100.0
    return np.clip(vals, 0.0, 1.0)


def run_episode(scenario, eval_seed: int, agent: DQNAgent) -> tuple[list[dict], bool, float]:
    """Drive one greedy episode; return (per-step rows, gridlock flag, final backlog fraction)."""
    import traci  # the env's own binding (libsumo when LIBSUMO_AS_TRACI=1)

    env = SUMOEnv(
        write_routes(scenario, eval_seed),
        episode_length_s=scenario.duration_s,
        sumo_seed=eval_seed,
        signal_mode="rl",
    )
    rows: list[dict] = []
    gridlocked, backlog_frac = False, 0.0
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            mask = info["mask"]
            q = _q_values(agent, obs)
            action = int(np.argmax(np.where(mask, q, _MASK_FILL)))

            exit_occ = _occupancy(traci, EXIT_EDGES)
            appr_occ = _occupancy(traci, APPROACH_EDGES)
            pending = len(traci.simulation.getPendingVehicles())
            queue = float(env.movement_features()[0].sum())

            row: dict[str, Any] = {
                "step": step,
                "sim_time": float(info.get("sim_time", step * 10)),
                "pending": pending,
                "departed": int(env.departed_count),
                "total_queue": queue,
                "exit_occ_mean": float(exit_occ.mean()),
                "exit_occ_max": float(exit_occ.max()),
                "appr_occ_mean": float(appr_occ.mean()),
                "appr_occ_max": float(appr_occ.max()),
            }
            row.update({f"exit_{e}": float(v) for e, v in zip(EXIT_EDGES, exit_occ)})
            row.update({f"appr_{e}": float(v) for e, v in zip(APPROACH_EDGES, appr_occ)})
            rows.append(row)

            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        ep = info.get("episode", {})
        backlog_frac = float(ep.get("insertion_backlog_fraction", 0.0))
        gridlocked = backlog_frac > _GRIDLOCK_BACKLOG
    finally:
        env.close()
    return rows, gridlocked, backlog_frac


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing rolling mean of window ``w`` (element i = mean of x[max(0,i-w+1)..i])."""
    if len(x) == 0:
        return x
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1) : i + 1].mean()
    return out


def onset_step(pending: np.ndarray) -> int | None:
    """First decision step at which insertion backlog becomes sustained (gridlock onset).

    Onset = the first index at which EVERY step of the trailing ``ONSET_WINDOW`` has at least
    ``ONSET_PENDING`` vehicles pending insertion. A trailing *mean* was rejected here: one
    extreme spike can drag a mean over the line, which would report onset for a momentary
    insertion hiccup rather than for a backlog that persists. Returns None if never reached.
    """
    x = pending.astype(np.float64)
    if len(x) < ONSET_WINDOW:
        return None
    for i in range(ONSET_WINDOW - 1, len(x)):
        if np.all(x[i - ONSET_WINDOW + 1 : i + 1] >= ONSET_PENDING):
            return i
    return None


def episode_summary(rows: list[dict]) -> dict[str, Any]:
    """Reduce one episode to the quantities the pre-registered verdict rule consumes."""
    exit_mean = np.array([r["exit_occ_mean"] for r in rows])
    exit_max = np.array([r["exit_occ_max"] for r in rows])
    appr_mean = np.array([r["appr_occ_mean"] for r in rows])
    pending = np.array([r["pending"] for r in rows])

    onset = onset_step(pending)
    sustained_exit = _rolling_mean(exit_mean, SUSTAIN_WINDOW)
    sustained_exit_max = _rolling_mean(exit_max, SUSTAIN_WINDOW)
    pre = slice(0, onset + 1) if onset is not None else slice(0, len(rows))
    return {
        "onset_step": onset,
        "backlog_grew": bool(pending[-1] > pending[0] or (onset is not None)),
        "exit_sustained_pre_onset": float(sustained_exit[pre].max()) if len(rows) else 0.0,
        "exit_max_sustained_pre_onset": float(sustained_exit_max[pre].max()) if len(rows) else 0.0,
        "exit_peak_any": float(exit_max.max()) if len(rows) else 0.0,
        "appr_peak_any": float(appr_mean.max()) if len(rows) else 0.0,
        "pending_final": int(pending[-1]) if len(rows) else 0,
        "pending_peak": int(pending.max()) if len(rows) else 0,
    }


def verdict(summaries: list[dict]) -> tuple[str, str]:
    """Apply the pre-registered rule to the gridlocked episodes. Returns (verdict, reasoning)."""
    grid = [s for s in summaries if s["gridlocked"]]
    if not grid:
        return "NO-GRIDLOCK", (
            "No episode in this run gridlocked, so the spillback-vs-backlog question cannot be "
            "answered from it. Not evidence for either branch."
        )
    exit_sus = np.array([s["exit_sustained_pre_onset"] for s in grid])
    exit_sus_max = np.array([s["exit_max_sustained_pre_onset"] for s in grid])
    grew = np.array([s["backlog_grew"] for s in grid])

    frac_attempt = float((exit_sus >= ATTEMPT_AT).mean())
    frac_dead = float((exit_sus < DEAD_BELOW).mean())

    if frac_attempt >= 0.5:
        return "ATTEMPT-WARRANTED", (
            f"{frac_attempt:.0%} of gridlocked episodes hold mean exit occupancy >= {ATTEMPT_AT:.2f} "
            f"sustained over {SUSTAIN_WINDOW * 10} s BEFORE onset (median "
            f"{np.median(exit_sus):.3f}, worst-edge median {np.median(exit_sus_max):.3f}). "
            "Exit-edge spillback is real here -> ONE timeboxed (<=1 day) exit-occupancy mask attempt "
            "with threshold sweep {0.7, 0.8, 0.9}, per-scenario x per-seed vs sel/plain."
        )
    if frac_dead >= 0.5 and grew.mean() >= 0.5:
        return "DEAD", (
            f"{frac_dead:.0%} of gridlocked episodes never reach mean exit occupancy {DEAD_BELOW:.2f} "
            f"(median sustained {np.median(exit_sus):.3f}, worst-edge median "
            f"{np.median(exit_sus_max):.3f}) while insertion backlog grows in "
            f"{grew.mean():.0%} of them. The exits are NOT full: an exit-occupancy mask would never "
            "bind -> non-binding constraint = no-op (scar, struck x2). Candidate DEAD; sel/plain "
            "ships unchanged and this measurement becomes report evidence that gridlock is intrinsic."
        )
    return "INCONCLUSIVE", (
        f"Exit occupancy sits between the pre-registered thresholds (median sustained "
        f"{np.median(exit_sus):.3f}; {frac_attempt:.0%} of episodes >= {ATTEMPT_AT:.2f}, "
        f"{frac_dead:.0%} < {DEAD_BELOW:.2f}). Per finish-plan.md §Phase 0 the question CLOSES "
        "either way: no further gridlock work, sel/plain ships unchanged."
    )


def _plot(all_rows: list[dict], summaries: list[dict]) -> None:
    """One figure: exit vs approach occupancy and insertion backlog, aligned on gridlock onset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_g, ax_c) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, want_grid, title in (
        (ax_g, True, "GRIDLOCKED episodes"),
        (ax_c, False, "CLEAN (non-gridlocked) episodes"),
    ):
        keys = [(s["train_seed"], s["eval_seed"]) for s in summaries if s["gridlocked"] == want_grid]
        ax2 = ax.twinx()
        for ts, es in keys:
            rows = [r for r in all_rows if r["train_seed"] == ts and r["eval_seed"] == es]
            if not rows:
                continue
            t = np.array([r["sim_time"] for r in rows]) / 60.0
            ax.plot(t, [r["exit_occ_mean"] for r in rows], color="tab:red", alpha=0.45, lw=1.2)
            ax.plot(t, [r["appr_occ_mean"] for r in rows], color="tab:blue", alpha=0.45, lw=1.2)
            ax2.plot(t, [r["pending"] for r in rows], color="tab:green", alpha=0.35, lw=1.0, ls="--")
            s = next(x for x in summaries if x["train_seed"] == ts and x["eval_seed"] == es)
            if s["onset_step"] is not None and s["onset_step"] < len(t):
                ax.axvline(t[s["onset_step"]], color="k", alpha=0.25, lw=0.8)
        ax.axhline(DEAD_BELOW, color="tab:red", ls=":", lw=1.2)
        ax.axhline(ATTEMPT_AT, color="tab:red", ls="-.", lw=1.2)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("edge occupancy (fraction)")
        ax2.set_ylabel("vehicles pending insertion", color="tab:green")
        ax.set_title(f"SCN-05, plain-DQN ep299 — {title}  (n={len(keys)} episodes)")
    ax_c.set_xlabel("simulation time (min)")
    ax_g.plot([], [], color="tab:red", label="exit-edge occupancy (mean of 4)")
    ax_g.plot([], [], color="tab:blue", label="approach-edge occupancy (mean of 4)")
    ax_g.plot([], [], color="tab:green", ls="--", label="insertion backlog (right axis)")
    ax_g.plot([], [], color="k", alpha=0.3, label="gridlock onset")
    ax_g.axhline(ATTEMPT_AT, color="tab:red", ls="-.", lw=1.2, label=f"attempt threshold {ATTEMPT_AT}")
    ax_g.axhline(DEAD_BELOW, color="tab:red", ls=":", lw=1.2, label=f"dead threshold {DEAD_BELOW}")
    ax_g.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(_OUT_PLOT, dpi=150)
    plt.close(fig)


def _replay_from_csv() -> None:
    """Rebuild summaries, plot and verdict from the committed per-step CSV (no SUMO)."""
    _float_cols = {"sim_time", "total_queue", "backlog_frac"}
    rows: list[dict] = []
    with _OUT_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in r.items():
                if k.startswith(("exit_", "appr_")) or k in _float_cols:
                    row[k] = float(v)
                else:
                    row[k] = int(v)
            rows.append(row)

    summaries = []
    by_ep: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        by_ep.setdefault((r["train_seed"], r["eval_seed"]), []).append(r)
    for (ts, es), ep_rows in by_ep.items():
        s = episode_summary(ep_rows)
        s.update({
            "train_seed": ts,
            "eval_seed": es,
            "gridlocked": bool(ep_rows[0]["gridlocked"]),
            "backlog_frac": float(ep_rows[0].get("backlog_frac", float("nan"))),
        })
        summaries.append(s)
        print(f"{ts:>6} {es:>6} {'YES' if s['gridlocked'] else 'no':>5} "
              f"{str(s['onset_step']):>6} {s['exit_sustained_pre_onset']:8.3f} "
              f"{s['exit_max_sustained_pre_onset']:11.3f} {s['exit_peak_any']:9.3f} "
              f"{s['appr_peak_any']:9.3f} {s['pending_peak']:9d}", flush=True)
    _emit(rows, summaries)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenario", default="SCN-05")
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002, 7003, 7004])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument(
        "--from-csv",
        action="store_true",
        help="Re-derive summaries/plot/verdict from an existing diag_exit_occupancy.csv instead of "
        "re-simulating. The per-step CSV is the raw measurement; everything else is derived from "
        "it, so a change to the verdict logic can be re-applied without burning another SUMO run.",
    )
    args = p.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        _replay_from_csv()
        return

    build_net()
    scn = load_scenario(SCENARIO_DIR / f"scn_{args.scenario.split('-')[1]}.yaml")
    agents = {s: _load_plain(s) for s in args.seeds}

    all_rows: list[dict] = []
    summaries: list[dict] = []
    print(f"{'tseed':>6} {'eseed':>6} {'grid':>5} {'backlog':>8} {'onset':>6} "
          f"{'exitSus':>8} {'exitMaxSus':>11} {'exitPeak':>9} {'apprPeak':>9} {'pendPeak':>9}",
          flush=True)
    print("-" * 96, flush=True)
    for ts, agent in agents.items():
        for es in args.eval_seeds:
            rows, grid, frac = run_episode(scn, es, agent)
            for r in rows:
                r["train_seed"], r["eval_seed"], r["gridlocked"] = ts, es, int(grid)
                r["backlog_frac"] = frac
            all_rows.extend(rows)
            s = episode_summary(rows)
            s.update({"train_seed": ts, "eval_seed": es, "gridlocked": grid, "backlog_frac": frac})
            summaries.append(s)
            print(f"{ts:>6} {es:>6} {'YES' if grid else 'no':>5} {frac:8.3f} "
                  f"{str(s['onset_step']):>6} {s['exit_sustained_pre_onset']:8.3f} "
                  f"{s['exit_max_sustained_pre_onset']:11.3f} {s['exit_peak_any']:9.3f} "
                  f"{s['appr_peak_any']:9.3f} {s['pending_peak']:9d}", flush=True)

    if all_rows:
        cols = list(all_rows[0].keys())
        with _OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
    _emit(all_rows, summaries)


def _emit(all_rows: list[dict], summaries: list[dict]) -> None:
    """Render the plot + verdict markdown from per-step rows and their episode summaries."""
    if all_rows:
        _plot(all_rows, summaries)

    v, why = verdict(summaries)
    n_grid = sum(1 for s in summaries if s["gridlocked"])
    lines = [
        "# Phase-0 diagnostic — exit-edge spillback vs insertion backlog (SCN-05, plain-DQN ep299)",
        "",
        f"**VERDICT: {v}**",
        "",
        why,
        "",
        f"- Episodes: {len(summaries)} ({n_grid} gridlocked at backlog > {_GRIDLOCK_BACKLOG}).",
        f"- Pre-registered thresholds (fixed in code before the first run, finish-plan.md §Phase 0): "
        f"dead < {DEAD_BELOW}, attempt >= {ATTEMPT_AT}, sustained over {SUSTAIN_WINDOW * 10} s; "
        f"onset = {ONSET_PENDING}+ vehicles pending for {ONSET_WINDOW * 10} s.",
        f"- Raw per-step data: `{_OUT_CSV.relative_to(_REPO_ROOT)}` · plot: "
        f"`{_OUT_PLOT.relative_to(_REPO_ROOT)}`",
        "",
        "| train_seed | eval_seed | gridlock | backlog_frac | onset_step | exit_sustained_pre_onset "
        "| exit_worst_edge_sustained | exit_peak | approach_peak | pending_peak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['train_seed']} | {s['eval_seed']} | {'YES' if s['gridlocked'] else 'no'} | "
            f"{s['backlog_frac']:.3f} | {s['onset_step']} | {s['exit_sustained_pre_onset']:.3f} | "
            f"{s['exit_max_sustained_pre_onset']:.3f} | {s['exit_peak_any']:.3f} | "
            f"{s['appr_peak_any']:.3f} | {s['pending_peak']} |"
        )
    _OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nVERDICT: {v}\n{why}")
    print(f"\nwrote {_OUT_CSV}\n      {_OUT_PLOT}\n      {_OUT_MD}")


if __name__ == "__main__":
    main()
