"""T-02-09 - Throughput pilot: is the full training matrix feasible overnight?

Measures, on *this* box:

* SUMO env decision-steps/sec under both the **TraCI** socket backend and the
  in-process **libsumo** backend (``LIBSUMO_AS_TRACI=1``) - libsumo is the one
  training would use;
* DQN **gradient-steps/sec** for the locked 56->128->128->8 MLP (torch);

then projects the wall-clock for the locked matrix - 300 episodes x 4 variants x 3
seeds, 360 decision steps/episode - and flags if it exceeds the 36 h escalation
bar (backlog T-02-09 / open-items E2). This is a hard prerequisite of T-03-07:
run it before queuing the overnight runs.

The two env backends are each timed in a **subprocess** so the ``traci`` vs
``libsumo`` import is clean (the choice is import-time, set by an env var). The
gradient timing runs in-process (it never touches SUMO).

Run::

    python -m scripts.benchmark_env                 # full pilot + report
    python -m scripts.benchmark_env --env-only --json   # internal: one backend
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _REPO_ROOT / "data" / "benchmark_report.json"

# Locked training matrix (training-infrastructure.md / evaluation-methodology.md).
EPISODES = 300
VARIANTS = 4  # DQN-forecast, DQN-no-forecast, DQN-random-LSTM, Double-DQN
SEEDS = 3
STEPS_PER_EP = 360  # 3600 s / 10 s decision interval
ESCALATION_HOURS = 36.0

_BENCH_SCENARIO = "SCN-02"  # heavy -> realistic worst-case per-step cost
_GRAD_BATCH = 64


def benchmark_env(n_steps: int, *, scenario_id: str = _BENCH_SCENARIO) -> dict:
    """Time ``n_steps`` of ``env.step`` (excluding one-time SUMO startup)."""
    scn = load_scenario(_REPO_ROOT / "config" / "scenarios" / f"scn_{scenario_id.split('-')[1]}.yaml")
    route = write_routes(scn, 0)
    # episode long enough that the heavy scenario never empties over n_steps
    env = SUMOEnv(route, episode_length_s=(n_steps + 50) * 10)
    try:
        obs, info = env.reset()  # startup cost lives here, outside the timed loop
        done = False
        t0 = time.perf_counter()
        for i in range(n_steps):
            if done:
                env.reset()
                done = False
            _, _, terminated, truncated, _ = env.step(i % 8)
            done = terminated or truncated
        elapsed = time.perf_counter() - t0
    finally:
        env.close()
    return {"n_steps": n_steps, "elapsed_s": elapsed, "steps_per_sec": n_steps / elapsed}


def benchmark_gradient(n_steps: int, *, batch: int = _GRAD_BATCH) -> dict:
    """Time ``n_steps`` of forward+backward+step on the locked DQN MLP (CPU/GPU)."""
    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(
        nn.Linear(56, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    x = torch.randn(batch, 56, device=device)
    y = torch.randn(batch, 8, device=device)

    for _ in range(5):  # warm up (lazy init, cudnn autotune)
        opt.zero_grad(); loss_fn(net(x), y).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad(); loss_fn(net(x), y).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"n_steps": n_steps, "elapsed_s": elapsed, "steps_per_sec": n_steps / elapsed, "device": device}


def project_training(env_sps: float, grad_sps: float) -> dict:
    """Project the matrix wall-clock from measured env + gradient rates."""
    total_env = EPISODES * VARIANTS * SEEDS * STEPS_PER_EP
    total_grad = total_env  # ~1 gradient update per env step after warm-up
    env_h = total_env / env_sps / 3600.0
    grad_h = total_grad / grad_sps / 3600.0 if grad_sps else 0.0
    total_h = env_h + grad_h
    return {
        "total_env_steps": total_env,
        "env_hours": env_h,
        "grad_hours": grad_h,
        "total_hours": total_h,
        "exceeds_escalation": total_h > ESCALATION_HOURS,
    }


def _run_env_backend(mode: str, n_steps: int) -> dict | None:
    """Run ``benchmark_env`` for one backend in a clean subprocess; parse its JSON."""
    env = dict(os.environ)
    if mode == "libsumo":
        env["LIBSUMO_AS_TRACI"] = "1"
    else:
        env.pop("LIBSUMO_AS_TRACI", None)
    cmd = [sys.executable, "-m", "scripts.benchmark_env", "--env-only",
           "--n-steps", str(n_steps), "--json"]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.splitlines()):  # last JSON line is the result
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def _fmt(res: dict | None, label: str) -> str:
    if res is None:
        return f"  {label:<10s} UNAVAILABLE"
    return f"  {label:<10s} {res['steps_per_sec']:8.1f} steps/s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-only", action="store_true", help="(internal) time one env backend")
    parser.add_argument("--json", action="store_true", help="(internal) emit JSON only")
    parser.add_argument("--env-steps", type=int, default=500, help="env decision steps to time")
    parser.add_argument("--grad-steps", type=int, default=100, help="gradient steps to time")
    parser.add_argument("--n-steps", type=int, default=500, help="(internal) steps for --env-only")
    args = parser.parse_args()

    if args.env_only:  # subprocess path: time this process's backend, print JSON
        print(json.dumps(benchmark_env(args.n_steps)))
        return

    build_net()
    traci_res = _run_env_backend("traci", args.env_steps)
    libsumo_res = _run_env_backend("libsumo", args.env_steps)
    grad_res = benchmark_gradient(args.grad_steps)

    # project with the backend training would actually use (libsumo if available)
    chosen = libsumo_res or traci_res
    projection = project_training(chosen["steps_per_sec"], grad_res["steps_per_sec"])

    report = {
        "env_traci": traci_res,
        "env_libsumo": libsumo_res,
        "gradient": grad_res,
        "projection_backend": "libsumo" if libsumo_res else "traci",
        "projection": projection,
        "matrix": {"episodes": EPISODES, "variants": VARIANTS, "seeds": SEEDS,
                   "steps_per_episode": STEPS_PER_EP},
    }
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== T-02-09 throughput pilot ===")
    print("env decision-steps/sec:")
    print(_fmt(traci_res, "traci"))
    print(_fmt(libsumo_res, "libsumo"))
    print(f"  gradient   {grad_res['steps_per_sec']:8.1f} steps/s  ({grad_res['device']})")
    speedup = (libsumo_res["steps_per_sec"] / traci_res["steps_per_sec"]
               if libsumo_res and traci_res else None)
    if speedup:
        print(f"  libsumo speedup: {speedup:.1f}x")
    print(f"\nmatrix: {EPISODES} ep x {VARIANTS} variants x {SEEDS} seeds "
          f"= {projection['total_env_steps']:,} env steps")
    print(f"projected wall-clock ({report['projection_backend']}): "
          f"{projection['total_hours']:.1f} h "
          f"(env {projection['env_hours']:.1f} h + grad {projection['grad_hours']:.2f} h)")
    verdict = ("ESCALATE - exceeds 36 h, cut scope before overnight runs"
               if projection["exceeds_escalation"]
               else f"OK - under the {ESCALATION_HOURS:.0f} h bar")
    print(f"verdict: {verdict}")
    print(f"\nreport -> {_REPORT.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
