"""T-03-08 - Single-seed sanity gate: the hard go/no-go before the full training matrix.

Runs 1 seed x 1 scenario (SCN-01) x 100 episodes through the real training loop, then
inspects the three things that catch a silently-broken DQN *before* it costs the full
T-03-07 matrix (3 seeds x 4 variants) and only surfaces at T-04-01:

1. **Reward trends up** - the learning curve improves from the (high-epsilon) start to the
   (exploitation) end. Compared as first-quartile vs last-quartile mean episode reward, plus
   a least-squares slope over all episodes.
2. **Q-values bounded** - the online Q stays finite and does not explode (the classic
   "reward not clipped / loss blows up" failure in training-infrastructure.md's failure table).
3. **Mask fires at the expected rate** - the action mask is actually constraining choices
   (min-green hold / max-green force), not silently all-ones. Read from the per-episode mean
   legal-action count (8 = never constrained).

Writes a ``verdict.md`` note next to the run and exits 0 (GO) / 1 (NO-GO). The verdict is a
reasoned heuristic with every number printed - the human makes the final call, per the DoD.

Run::

    python -m scripts.sanity_gate                       # SCN-01, seed 42, 100 episodes
    python -m scripts.sanity_gate --seed 123
    python -m scripts.sanity_gate --episodes 5 --episode-length 200   # quick self-test
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

from scripts.build_network import build_net
from scripts.env_factory import build_env
from src.ml.hybrid_wrapper import load_forecaster
from src.ml.train_loop import TrainConfig, train
from src.provenance.versions import git_sha

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"

# Loose guards - they flag obvious pathologies, not fine-grained quality (that is T-04).
_Q_EXPLODE_ABS = 1.0e5  # |Q| beyond this (or non-finite) = exploding value function
_MASK_FIRE_MIN = 0.01   # at least this fraction of steps must be constrained, else mask is dead


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def _col(header: list[str], rows: list[list[str]], name: str) -> list[float]:
    """Pull one numeric column by header name, dropping empty cells (pre-warmup rows)."""
    i = header.index(name)
    out: list[float] = []
    for r in rows:
        if i < len(r) and r[i] != "":
            out.append(float(r[i]))
    return out


def _analyze(run_dir: Path) -> tuple[bool, list[str], dict[str, float]]:
    """Return ``(go, reasons, metrics)`` from the run's episode/step CSVs."""
    ep_header, ep_rows = _read_csv(run_dir / "episodes.csv")
    st_header, st_rows = _read_csv(run_dir / "steps.csv")

    rewards = np.asarray(_col(ep_header, ep_rows, "ep_reward"))
    legal_means = np.asarray(_col(ep_header, ep_rows, "mask_legal_mean"))
    q_maxes = np.asarray(_col(st_header, st_rows, "q_max")) if st_rows else np.asarray([])
    q_means = np.asarray(_col(st_header, st_rows, "q_mean")) if st_rows else np.asarray([])

    n = len(rewards)
    qtile = max(1, n // 4)
    first_q = float(rewards[:qtile].mean())
    last_q = float(rewards[-qtile:].mean())
    slope = float(np.polyfit(np.arange(n), rewards, 1)[0]) if n >= 2 else 0.0

    q_abs_max = float(np.nanmax(np.abs(q_maxes))) if q_maxes.size else 0.0
    q_finite = bool(np.isfinite(q_maxes).all() and np.isfinite(q_means).all()) if q_maxes.size else True
    mean_legal = float(legal_means.mean()) if legal_means.size else 8.0
    fire_rate = (8.0 - mean_legal) / 8.0  # fraction of legal actions removed on average

    metrics = {
        "episodes": float(n),
        "reward_first_quartile_mean": first_q,
        "reward_last_quartile_mean": last_q,
        "reward_slope_per_episode": slope,
        "q_abs_max": q_abs_max,
        "mean_legal_actions": mean_legal,
        "mask_fire_rate": fire_rate,
        "learn_steps_logged": float(q_maxes.size),
    }

    reasons: list[str] = []
    trends_up = last_q >= first_q
    reasons.append(
        f"{'PASS' if trends_up else 'FAIL'}: reward trend - last-quartile mean {last_q:.1f} "
        f"vs first-quartile {first_q:.1f} (slope {slope:+.2f}/ep)."
    )
    q_ok = q_finite and q_abs_max < _Q_EXPLODE_ABS
    reasons.append(
        f"{'PASS' if q_ok else 'FAIL'}: Q bounded - max|Q|={q_abs_max:.1f} "
        f"(finite={q_finite}, limit {_Q_EXPLODE_ABS:.0e})."
        + ("" if q_maxes.size else " [no learn steps logged - run too short to judge]")
    )
    mask_ok = fire_rate >= _MASK_FIRE_MIN
    reasons.append(
        f"{'PASS' if mask_ok else 'FAIL'}: mask active - mean {mean_legal:.2f}/8 legal "
        f"actions (fire-rate {fire_rate:.1%})."
    )

    go = trends_up and q_ok and mask_ok
    return go, reasons, metrics


def _write_verdict(run_dir: Path, go: bool, reasons: list[str], metrics: dict[str, float], cfg: TrainConfig) -> Path:
    lines = [
        f"# T-03-08 sanity-gate verdict - {'GO' if go else 'NO-GO'}",
        "",
        f"- generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"- scenario: {cfg.train_scenarios[0]} | seed: {cfg.seed} | episodes: {cfg.n_episodes}"
        f" | variant: {cfg.variant} | git: {cfg.git_sha or 'n/a'}",
        "",
        "## Checks",
        *[f"- {r}" for r in reasons],
        "",
        "## Metrics",
        *[f"- {k}: {v:.4g}" for k, v in metrics.items()],
        "",
        "_Heuristic gate; numbers above are the evidence - confirm the learning curve in "
        "`episodes.csv` before queuing T-03-07._",
        "",
    ]
    path = run_dir / "verdict.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default="SCN-01", help="single scenario id (default SCN-01)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=None, help="override (for self-test)")
    parser.add_argument("--switch-penalty", type=float, default=0.1,
                        help="reward switch penalty lambda (default 0.1)")
    parser.add_argument("--forecast-ckpt", default=None, help="optional: gate the hybrid variant")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    variant = "hybrid" if args.forecast_ckpt else "plain"
    run_dir = (Path(args.run_dir) if args.run_dir
               else _RUNS_DIR / f"sanity_{args.scenario}_seed{args.seed}").resolve()

    cfg = TrainConfig(
        variant=variant,
        seed=args.seed,
        n_episodes=args.episodes,
        episode_length_s=args.episode_length or 3600,
        train_scenarios=(args.scenario,),  # single scenario - no rotation (the gate's point)
        validation_every=0,                # gate judges training dynamics, not held-out reward
        checkpoint_every=50,
        switch_penalty=args.switch_penalty,
        forecast_ckpt=args.forecast_ckpt,
        log_steps=True,                    # Q stats per learn step feed the bounded-Q check
        git_sha=git_sha(short=True) or "",
    )

    build_net()
    forecaster = load_forecaster(args.forecast_ckpt) if args.forecast_ckpt else None

    def make_env(scenario_id: str, route_seed: int):
        return build_env(
            scenario_id, route_seed, forecaster=forecaster,
            episode_length_s=args.episode_length, switch_penalty=cfg.switch_penalty,
        )

    print(f"[sanity] {args.scenario} seed={args.seed} episodes={cfg.n_episodes} "
          f"variant={variant} -> {run_dir}")
    train(
        cfg,
        make_train_env=make_env,
        make_val_env=lambda rs: make_env(args.scenario, rs),
        run_dir=run_dir,
    )

    go, reasons, metrics = _analyze(run_dir)
    verdict_path = _write_verdict(run_dir, go, reasons, metrics, cfg)

    print(f"\n[sanity] === {'GO' if go else 'NO-GO'} ===")
    for r in reasons:
        print(f"[sanity]   {r}")
    try:
        shown = verdict_path.relative_to(_REPO_ROOT)
    except ValueError:
        shown = verdict_path
    print(f"[sanity] verdict -> {shown}")
    sys.exit(0 if go else 1)


if __name__ == "__main__":
    main()
