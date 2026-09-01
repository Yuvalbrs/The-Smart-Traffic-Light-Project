"""Apply the PRE-COMMITTED retrain rule (decisions.md 2026-09-01) to the seed runs.

The rule was written before the runs existed (commit ``bf7e6fb``): train the forecaster on the
three canonical seeds {42, 123, 2024}, touch no other knob, and **apply the freeze gate to the
MEAN skill score across the three seeds, not to the best one** - "picking the best of three is a
maximum over noise and would manufacture skill".

The three runs exist and their per-seed reports are on disk, but the rule's own verdict - the
mean-gate - had never been computed or recorded anywhere. A pre-commitment that is never cashed
out is not a pre-commitment. This script cashes it out and writes the artefact, so the number
that decides H2's fate is reproducible rather than pasted.

It reads only the per-seed reports; it trains nothing and re-pins nothing.

Run::

    python -m scripts.forecaster_gate_report
    python -m scripts.forecaster_gate_report --data-version data-222f7d7e77a3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.ml.lstm_data import SPLITS
from src.ml.lstm_model import HORIZON, gate_verdict
from scripts.train_lstm import _dataset_data_version

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CKPT_DIR = _REPO_ROOT / "checkpoints" / "lstm"
_DATA_DIR = _REPO_ROOT / "data" / "lstm"

# The seed set fixed by the retrain rule BEFORE any of the runs existed. Hard-coded, not
# discovered from whatever reports happen to be lying around: a rule that adapts to the files
# on disk is a rule that can be satisfied by deleting a file.
PRECOMMITTED_SEEDS: tuple[int, ...] = (42, 123, 2024)


# The pre-A6 split, reconstructed. Reports written before A6.4 carry no ``splits`` field, and
# the corpus hash cannot tell the two splits apart - it covers all 100 CSVs whichever subset is
# read - so "no splits field" IS the identifier of the pre-split runs. Spelled out here rather
# than inferred, because a missing field silently matching everything is how a merge happens.
LEGACY_SPLITS: dict[str, list[str]] = {
    "test": ["SCN-05"], "train": ["SCN-01", "SCN-02", "SCN-03"], "val": ["SCN-04"],
}


def _reports_for(data_version: str, splits: dict[str, list[str]]) -> dict[int, dict]:
    """``{seed: report}`` for reports trained on ``data_version`` AND on exactly ``splits``."""
    out: dict[int, dict] = {}
    for path in sorted(_CKPT_DIR.glob("lstm-*__report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("data_version") != data_version:
            continue
        if report.get("splits", LEGACY_SPLITS) != splits:
            continue
        seed = report["seed"]
        if seed in out:
            raise RuntimeError(
                f"two reports for seed {seed} on {data_version}: {out[seed]['lstm_version']} and "
                f"{report['lstm_version']}. The rule averages one run per declared seed; pick one "
                f"and move the other out of checkpoints/lstm/ before re-running."
            )
        out[seed] = report
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-version", default=None,
        help="corpus id to score (default: the corpus currently in data/lstm/)",
    )
    parser.add_argument(
        "--split", choices=("current", "legacy"), default="current",
        help="which split's runs to score: 'current' = src/ml/lstm_data.py::SPLITS (A6.2, the "
             "declared SECONDARY analysis); 'legacy' = the pre-A6 train 01/02/03 split (the "
             "pre-registered PRIMARY, preregistration.md A6.3.1)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    splits = (
        {k: list(v) for k, v in sorted(SPLITS.items())} if args.split == "current"
        else LEGACY_SPLITS
    )
    status = "SECONDARY (declared post-hoc, A6.3.2)" if args.split == "current"         else "PRIMARY (pre-registered, A6.3.1)"
    out_path = Path(args.out) if args.out else _CKPT_DIR / f"retrain_rule_verdict__{args.split}.json"

    data_version = args.data_version or _dataset_data_version(_DATA_DIR)
    reports = _reports_for(data_version, splits)

    missing = [s for s in PRECOMMITTED_SEEDS if s not in reports]
    if missing:
        raise SystemExit(
            f"the retrain rule declares seeds {list(PRECOMMITTED_SEEDS)}; no report on "
            f"{data_version} for seed(s) {missing}. Train them before reading a verdict - the mean "
            f"of the seeds that happened to finish is exactly the best-of-N the rule forbids."
        )
    extra = sorted(set(reports) - set(PRECOMMITTED_SEEDS))

    per_seed = [reports[s] for s in PRECOMMITTED_SEEDS]  # declared order, not disk order
    n = len(per_seed)
    mean_ss = [sum(r["skill_scores_val"][h] for r in per_seed) / n for h in range(HORIZON)]
    mean_r2 = sum(r["r2_val"] for r in per_seed) / n
    decision = gate_verdict(mean_ss[0], mean_ss[HORIZON - 1])

    verdict = {
        "rule": "decisions.md 2026-09-01 FORECASTER RETRAIN RULE (pre-committed, commit bf7e6fb)",
        "evidential_status": status,
        "splits": splits,
        "data_version": data_version,
        "declared_seeds": list(PRECOMMITTED_SEEDS),
        "ignored_seeds_present_on_disk": extra,
        "per_seed": [
            {
                "seed": r["seed"], "lstm_version": r["lstm_version"], "checkpoint": r["checkpoint"],
                "skill_scores_val": r["skill_scores_val"], "r2_val": r["r2_val"],
                "val_mse": r["val_mse"], "test_mse": r["test_mse"],
                "own_verdict": r["gate"]["verdict"],
            }
            for r in per_seed
        ],
        "mean_skill_scores_val": mean_ss,
        "mean_r2_val": mean_r2,
        "gate_on_mean": asdict(decision),
        # The gate consults ss[0] and ss[-1] only. Carried explicitly so a horizon that the
        # verdict never looked at cannot go unreported.
        "horizons_not_consulted_by_gate": {
            f"index_{h}": mean_ss[h] for h in range(1, HORIZON - 1)
        },
    }
    out_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    print(f"[gate] {status}")
    print(f"[gate] splits: " + " | ".join(f"{k}={','.join(v)}" for k, v in splits.items()))
    print(f"[gate] corpus {data_version}, declared seeds {list(PRECOMMITTED_SEEDS)}")
    for r in per_seed:
        ss = r["skill_scores_val"]
        print(f"  seed {r['seed']:>5}  {r['lstm_version']}  SS={ss[0]:+.4f}/{ss[1]:+.4f}/{ss[2]:+.4f}"
              f"  r2={r['r2_val']:+.3f}  own={r['gate']['verdict']}")
    print(f"[gate] MEAN SS = {mean_ss[0]:+.5f}/{mean_ss[1]:+.5f}/{mean_ss[2]:+.5f}   mean r2 = {mean_r2:+.4f}")
    for h in range(1, HORIZON - 1):
        print(f"[gate] NOT consulted by the gate: mean SS at horizon index {h} = {mean_ss[h]:+.5f}")
    print(f"[gate] PRE-COMMITTED VERDICT: {decision.verdict} (ship={decision.ship})")
    print(f"       {decision.reason}")
    print(f"[gate] OK -> {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
