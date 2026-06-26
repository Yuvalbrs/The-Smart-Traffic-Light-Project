"""Live dashboard for the training matrix (T-03-07). Pure reader - safe to run anytime.

Self-refreshing terminal view of all 9 cells (variant x seed): status, episodes X/300 with a
progress bar, the latest episode reward, and best-validation reward once a cell finishes. Reads
ONLY the files the matrix writes (``runs/matrix_summary.json`` + each
``runs/{variant}_seed{seed}/episodes.csv``); it never starts, stops, or touches training - so
you can open it in a second terminal and watch the run that is already going.

Run::

    python -m scripts.watch_matrix          # live, refresh every 2 s, until all done / Ctrl+C
    python -m scripts.watch_matrix --once    # print one snapshot and exit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

_RUNS = Path(__file__).resolve().parent.parent / "runs"
VARIANTS = ("plain", "hybrid", "random-lstm")
SEEDS = (42, 123, 2024)
TOTAL = 300


def _cell_progress(run_dir: Path) -> tuple[int, float | None]:
    """(#episodes written, latest ep_reward) from a cell's episodes.csv, or (0, None)."""
    f = run_dir / "episodes.csv"
    if not f.exists():
        return 0, None
    with f.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) <= 1:
        return 0, None
    header, last = rows[0], rows[-1]
    reward: float | None = None
    if "ep_reward" in header:
        try:
            reward = float(last[header.index("ep_reward")])
        except (ValueError, IndexError):
            reward = None
    return len(rows) - 1, reward


def _summary() -> dict[tuple[str, int], dict]:
    """Map (variant, seed) -> the matrix_summary.json record, if any."""
    sp = _RUNS / "matrix_summary.json"
    out: dict[tuple[str, int], dict] = {}
    if sp.exists():
        try:
            for r in json.loads(sp.read_text(encoding="utf-8")):
                out[(r["variant"], r["seed"])] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return out


def _bar(done: int, total: int, width: int = 22) -> str:
    filled = int(width * done / total) if total else 0
    return "#" * filled + "." * (width - filled)


def render() -> bool:
    """Print one snapshot; return True once every cell is finished (done or failed)."""
    summ = _summary()
    out = ["", "  ===== SMART-TRAFFIC-RL : TRAINING MATRIX (T-03-07) =====", ""]
    finished = 0
    for v in VARIANTS:
        for s in SEEDS:
            rd = _RUNS / f"{v}_seed{s}"
            rec = summ.get((v, s))
            done_eps, reward = _cell_progress(rd)
            status = str(rec.get("status", "")) if rec else ""
            if status in ("ok", "skipped-complete"):
                tag, done_eps, finished = "DONE  ", TOTAL, finished + 1
                bv = rec.get("best_val_reward")
                extra = f"best_val={bv:,.0f}" if isinstance(bv, (int, float)) else ""
            elif status.startswith("FAILED"):
                tag, finished = "FAILED", finished + 1
                extra = status[:44]
            elif done_eps > 0:
                tag = "train "
                extra = f"last_reward={reward:,.0f}" if reward is not None else ""
            else:
                tag, extra = "queued", ""
            out.append(f"  {v + '_seed' + str(s):18} [{_bar(done_eps, TOTAL)}] "
                       f"{done_eps:3d}/{TOTAL}  {tag}  {extra}")
    out += ["", f"  finished {finished}/9   "
            "(DONE = trained ok | train = running now | queued = not started | FAILED = error)", ""]
    print("\n".join(out), flush=True)
    return finished == 9


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--once", action="store_true", help="one snapshot then exit")
    p.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default 2)")
    args = p.parse_args()

    if args.once:
        render()
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            if render():
                print("  >>> ALL 9 CELLS FINISHED. <<<")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  (stopped watching - training is NOT affected, it keeps running)")


if __name__ == "__main__":
    main()
