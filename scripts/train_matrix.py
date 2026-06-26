"""T-03-07 - Main training matrix: variants x seeds, one coherent batch.

Runs the locked mandatory variants across the 3 seeds {42, 123, 2024}, each a full
300-episode run into its own ``runs/{variant}_seed{seed}/`` with a ``config.yaml`` carrying
the git SHA (provenance). The three variants are the never-cut set:

* **plain**       - 20-dim DQN, no forecast.
* **hybrid**      - 56-dim DQN + the frozen trained forecaster (the headline idea).
* **random-lstm** - 56-dim DQN + a frozen UNTRAINED forecaster (the control: proves any gain
  comes from a *real* forecast, not just from having 36 extra inputs).

Resilience (backlog T-03-07 partial-failure protocol): a cell whose final checkpoint already
exists is **skipped**; a cell with a partial run is **resumed** from its latest checkpoint; a
crash in one cell is recorded and does **not** abort the rest. ``runs/matrix_summary.json`` is
rewritten after every cell so progress survives an interruption (or a context reset).

Run (use libsumo - the training backend - and run detached for the long batch)::

    LIBSUMO_AS_TRACI=1 python -m scripts.train_matrix
    LIBSUMO_AS_TRACI=1 python -m scripts.train_matrix --variants plain hybrid --seeds 42
    LIBSUMO_AS_TRACI=1 python -m scripts.train_matrix --episodes 5 --episode-length 200  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from scripts.build_network import build_net
from scripts.env_factory import build_env
from src.ml.hybrid_wrapper import load_forecaster, random_forecaster
from src.ml.train_loop import TrainConfig, train
from src.provenance.versions import git_sha

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"
_OFFICIAL_LSTM = (
    _REPO_ROOT / "checkpoints" / "lstm"
    / "lstm__data-8eb28eecdefb__lstm-df67afd839d4.pt"  # official ckpt (hot.md / ADR-006)
)
SEEDS = (42, 123, 2024)
VARIANTS = ("plain", "hybrid", "random-lstm")


def _forecaster_for(variant: str, seed: int):
    """Return ``(forecaster_or_None, provenance_label)`` for a variant."""
    if variant == "plain":
        return None, None
    if variant == "hybrid":
        if not _OFFICIAL_LSTM.exists():
            raise FileNotFoundError(
                f"hybrid variant needs the forecaster checkpoint at {_OFFICIAL_LSTM} "
                "(regenerate via scripts/train_lstm.py - it is gitignored)."
            )
        return load_forecaster(str(_OFFICIAL_LSTM)), _OFFICIAL_LSTM.name
    if variant == "random-lstm":
        return random_forecaster(seed=seed), "random-lstm"
    raise ValueError(f"unknown variant {variant!r}")


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """The highest-episode ep*.pt checkpoint in a run dir, or None."""
    ck_dir = run_dir / "checkpoints"
    if not ck_dir.exists():
        return None
    ckpts = [p for p in ck_dir.glob("ep*.pt")]
    return max(ckpts, key=lambda p: int(p.stem[2:])) if ckpts else None


def _run_cell(variant: str, seed: int, args: argparse.Namespace, sha: str) -> dict:
    """Run (or skip/resume) one matrix cell; return a summary record."""
    run_dir = (_RUNS_DIR / f"{variant}_seed{seed}").resolve()
    final_ckpt = run_dir / "checkpoints" / f"ep{args.episodes - 1}.pt"
    if final_ckpt.exists() and not args.force:
        return {"variant": variant, "seed": seed, "status": "skipped-complete",
                "run_dir": str(run_dir)}

    forecaster, fc_label = _forecaster_for(variant, seed)
    resume = None if args.force else _latest_checkpoint(run_dir)

    cfg = TrainConfig(
        variant=variant, seed=seed, n_episodes=args.episodes,
        episode_length_s=args.episode_length or 3600,
        switch_penalty=args.switch_penalty,
        forecast=(variant != "plain"), forecast_ckpt=fc_label,
        validation_every=args.validation_every,
        git_sha=sha,
    )

    def mk(scenario_id: str, route_seed: int):
        return build_env(scenario_id, route_seed, forecaster=forecaster,
                         episode_length_s=args.episode_length, switch_penalty=cfg.switch_penalty)

    result = train(
        cfg, make_train_env=mk,
        make_val_env=lambda rs: mk(cfg.val_scenario, rs),
        run_dir=run_dir, resume=resume,
    )
    return {
        "variant": variant, "seed": seed, "status": "ok", "run_dir": str(run_dir),
        "episodes_completed": result.episodes_completed, "total_steps": result.total_steps,
        "best_val_reward": result.best_val_reward,
        "resumed_from": str(resume) if resume else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=None, help="override (for smoke)")
    parser.add_argument("--switch-penalty", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="ignore existing checkpoints; rerun")
    args = parser.parse_args()

    build_net()
    sha = git_sha(short=True) or ""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _RUNS_DIR / "matrix_summary.json"

    summary: list[dict] = []
    cells = [(v, s) for v in args.variants for s in args.seeds]
    print(f"[matrix] {len(cells)} cells (variants={args.variants} seeds={args.seeds} "
          f"episodes={args.episodes}) git={sha}")
    for i, (variant, seed) in enumerate(cells, 1):
        print(f"\n[matrix] === cell {i}/{len(cells)}: {variant} seed={seed} ===")
        try:
            rec = _run_cell(variant, seed, args, sha)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the batch
            traceback.print_exc()
            rec = {"variant": variant, "seed": seed, "status": f"FAILED: {exc}",
                   "run_dir": str((_RUNS_DIR / f"{variant}_seed{seed}").resolve())}
        summary.append(rec)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[matrix] {variant} seed={seed} -> {rec['status']}")

    ok = sum(1 for r in summary if r["status"] in ("ok", "skipped-complete"))
    print(f"\n[matrix] DONE - {ok}/{len(cells)} cells ok -> {summary_path}")
    sys.exit(0 if ok == len(cells) else 1)


if __name__ == "__main__":
    main()
