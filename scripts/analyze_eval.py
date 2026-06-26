"""T-04-02 - Statistical analysis of the eval results, per the frozen pre-registration.

Reads ``data/eval/eval_results.csv`` (T-04-01) and runs the LOCKED analysis plan
(preregistration.md, frozen 2026-06-24): two-sided **Wilcoxon signed-rank**, paired, with the
**Pratt** zero treatment; **Holm-Bonferroni** within three SEPARATE confirmatory families on the
designated test scenario **SCN-05**; median paired difference + a bootstrap 95% CI as effect size.

Families (SCN-05):
  * C1 / H1 - hybrid vs each of {webster, max_pressure, actuated}, 7 KPIs -> 21 tests.
  * C2 / H2 - hybrid vs plain (the headline forecast ablation), 7 KPIs -> 7 tests.
  * C3 / H3 - hybrid vs random-lstm (information vs capacity), 7 KPIs -> 7 tests.

Pairing (preregistration s2): DQN-vs-DQN pairs by (train_seed, eval_seed) -> n=15; DQN-vs-baseline
pairs the baseline (no train seed) against each of hybrid's 3 train-seed models on the same
eval_seed -> n=15 (the baseline value repeats across train seeds; noted). Edge rules (s8): a pair
is dropped if BOTH algorithms are gridlock-censored for that KPI, or if either KPI is missing;
every drop is counted and reported.

The 7 KPIs + better-direction are the locked set (preregistration s3). SCN-01..04 are reported as
SUPPORTING/exploratory (raw p, no family correction) - this is where the regime-dependence shows.

Run::

    python -m scripts.analyze_eval                 # -> printed tables + data/eval/analysis.md + plots
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV = _REPO_ROOT / "data" / "eval" / "eval_results.csv"
_OUT_MD = _REPO_ROOT / "data" / "eval" / "analysis.md"
_PLOT_DIR = _REPO_ROOT / "data" / "eval" / "plots"

TRAIN_SEEDS = ("42", "123", "2024")
# (csv column, label, better-direction) - the locked 7-KPI confirmatory family.
KPIS = [
    ("avg_waiting_time", "avg_wait", "lower"),
    ("avg_queue_length", "avg_queue", "lower"),
    ("throughput", "throughput", "higher"),
    ("num_stops", "num_stops", "lower"),
    ("fairness_std", "fairness_std(5a)", "lower"),
    ("worst_movement_max_wait", "worst_max(5b)", "lower"),
    ("wait_p95", "wait_p95", "lower"),
]
HEADLINE = {"avg_waiting_time", "throughput", "worst_movement_max_wait"}
ALPHA = 0.05


def _load() -> list[dict]:
    return list(csv.DictReader(_CSV.open(encoding="utf-8")))


def _num(row: dict, col: str) -> float:
    v = row.get(col, "")
    if v in ("", "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _index(rows: list[dict]):
    """Index DQN rows by (variant, train_seed, scenario, eval_seed) and baselines by (algo, scn, seed)."""
    dqn: dict[tuple, dict] = {}
    base: dict[tuple, dict] = {}
    for r in rows:
        if r["variant"]:  # a DQN row
            dqn[(r["variant"], r["train_seed"], r["scenario"], r["eval_seed"])] = r
        else:
            base[(r["algo"], r["scenario"], r["eval_seed"])] = r
    return dqn, base


def _eval_seeds(rows: list[dict], scenario: str) -> list[str]:
    return sorted({r["eval_seed"] for r in rows if r["scenario"] == scenario}, key=int)


def _pairs(dqn, base, eval_seeds, scenario, kpi, a_variant, b_name, b_is_dqn):
    """Return (a_vals, b_vals, n_dropped) honoring the censoring + missing-pair rules."""
    a_vals, b_vals, dropped = [], [], 0
    for es in eval_seeds:
        for ts in TRAIN_SEEDS:
            ra = dqn.get((a_variant, ts, scenario, es))
            rb = (dqn.get((b_name, ts, scenario, es)) if b_is_dqn
                  else base.get((b_name, scenario, es)))  # baseline: same across train seeds
            if ra is None or rb is None:
                dropped += 1
                continue
            va, vb = _num(ra, kpi), _num(rb, kpi)
            both_censored = int(ra["gridlock_censored"]) and int(rb["gridlock_censored"])
            if both_censored or np.isnan(va) or np.isnan(vb):
                dropped += 1
                continue
            a_vals.append(va)
            b_vals.append(vb)
    return np.array(a_vals), np.array(b_vals), dropped


def _wilcoxon(a: np.ndarray, b: np.ndarray):
    """Two-sided Wilcoxon signed-rank (Pratt zeros). Returns (p, median_diff, ci_lo, ci_hi, n)."""
    d = a - b
    n = len(d)
    if n < 2 or np.allclose(d, 0.0):
        return float("nan"), float(np.median(d)) if n else float("nan"), float("nan"), float("nan"), n
    try:
        _, p = wilcoxon(d, zero_method="pratt", alternative="two-sided")
    except ValueError:
        p = float("nan")
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(d, size=n, replace=True)) for _ in range(2000)]
    return float(p), float(np.median(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs pass through, excluded from m)."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)  # enforce monotonicity
        adj[i] = running
    return adj


def _stars(p: float) -> str:
    if np.isnan(p):
        return "  "
    return "**" if p < 0.01 else ("* " if p < ALPHA else "  ")


def _family(dqn, base, eval_seeds, scenario, a_variant, comparisons, title, lines):
    """Run one Holm family (list of (b_name, b_is_dqn, label)) over the 7 KPIs; append a table."""
    results = []  # (kpi_label, b_label, direction, headline, p, med, lo, hi, n, dropped)
    for kpi, klabel, direction in KPIS:
        for b_name, b_is_dqn, blabel in comparisons:
            a, b, dropped = _pairs(dqn, base, eval_seeds, scenario, kpi, a_variant, b_name, b_is_dqn)
            p, med, lo, hi, n = _wilcoxon(a, b)
            results.append([klabel, blabel, direction, kpi in HEADLINE, p, med, lo, hi, n, dropped])
    adj = _holm([r[4] for r in results])
    lines.append(f"\n### {title}  (n target 15; Holm family size {sum(not np.isnan(r[4]) for r in results)})")
    lines.append("| KPI | vs | median d(hybrid-other) | 95% CI | n | drop | p_raw | p_holm | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r, pa in zip(results, adj):
        klabel, blabel, direction, headline, p, med, lo, hi, n, dropped = r
        sig = "YES" if (not np.isnan(pa) and pa < ALPHA) else ("." if not np.isnan(pa) else "n/a")
        hl = "**" if headline else ""
        ci = f"[{lo:.1f}, {hi:.1f}]" if not np.isnan(lo) else "—"
        pr = f"{p:.4f}" if not np.isnan(p) else "n/a"
        ph = f"{pa:.4f}{_stars(pa)}" if not np.isnan(pa) else "n/a"
        lines.append(f"| {hl}{klabel}{hl} | {blabel} | {med:+.2f} | {ci} | {n} | {dropped} | {pr} | {ph} | {sig} |")
    return results


def _supporting_regime(dqn, base, rows, lines):
    """H2 (hybrid vs plain) on the 3 headline KPIs across ALL scenarios - shows regime-dependence."""
    lines.append("\n## Supporting (exploratory, raw p): H2 hybrid−plain across scenarios")
    lines.append("_Where does the forecast help? (no family correction; SCN-05 is the confirmatory one above)_")
    lines.append("| scenario | KPI | median d(hybrid-plain) | n | drop | p_raw |")
    lines.append("|---|---|---|---|---|---|")
    for scenario in sorted({r["scenario"] for r in rows}, key=lambda s: s):
        es = _eval_seeds(rows, scenario)
        for kpi, klabel, _d in KPIS:
            if kpi not in HEADLINE:
                continue
            a, b, dropped = _pairs(dqn, base, es, scenario, kpi, "hybrid", "plain", True)
            p, med, lo, hi, n = _wilcoxon(a, b)
            pr = f"{p:.4f}" if not np.isnan(p) else "n/a"
            lines.append(f"| {scenario} | {klabel} | {med:+.2f} | {n} | {dropped} | {pr} |")


def _plots(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    groups = ["webster", "max_pressure", "actuated", "plain", "hybrid", "random-lstm"]

    def _key(r):
        return r["variant"] if r["variant"] else r["algo"]

    scenarios = sorted({r["scenario"] for r in rows})
    for col, label, plotname in [("throughput", "throughput veh/h (incl. censored)", "P_throughput"),
                                 ("avg_waiting_time", "avg wait s (valid only)", "P_wait")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.13
        x = np.arange(len(scenarios))
        for gi, g in enumerate(groups):
            means = []
            for sc in scenarios:
                vals = [_num(r, col) for r in rows if r["scenario"] == sc and _key(r) == g
                        and (col == "throughput" or not int(r["gridlock_censored"]))]
                vals = [v for v in vals if not np.isnan(v)]
                means.append(np.mean(vals) if vals else 0.0)
            ax.bar(x + gi * width, means, width, label=g)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by controller x scenario")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(_PLOT_DIR / f"{plotname}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console safety
    except (AttributeError, ValueError):
        pass
    rows = _load()
    dqn, base = _index(rows)
    es05 = _eval_seeds(rows, "SCN-05")
    lines = ["# T-04-02 Statistical Analysis (per frozen preregistration.md)",
             "",
             "Two-sided Wilcoxon signed-rank, Pratt zero-method, alpha=0.05. Effect size = median "
             "paired difference (hybrid - other) + bootstrap 95% CI. Holm-Bonferroni WITHIN each "
             "family. Decision bar = p_holm < 0.05. `**` headline KPI. Pair dropped if both "
             "gridlock-censored or a KPI is missing (drop count shown).",
             "",
             "## Confirmatory families - SCN-05 (designated test scenario)"]

    _family(dqn, base, es05, "SCN-05", "hybrid",
            [("webster", False, "webster"), ("max_pressure", False, "max_pressure"),
             ("actuated", False, "actuated")],
            "C1 / H1 - hybrid vs 3 baselines (21 tests)", lines)
    _family(dqn, base, es05, "SCN-05", "hybrid",
            [("plain", True, "plain")],
            "C2 / H2 - hybrid vs plain (forecast ablation, 7 tests)", lines)
    _family(dqn, base, es05, "SCN-05", "hybrid",
            [("random-lstm", True, "random-lstm")],
            "C3 / H3 - hybrid vs random-lstm (info vs capacity, 7 tests)", lines)

    _supporting_regime(dqn, base, rows, lines)

    try:
        _plots(rows)
        lines.append(f"\n_Plots: {_PLOT_DIR.relative_to(_REPO_ROOT)}/P_throughput.png, P_wait.png_")
    except Exception as exc:  # noqa: BLE001 - plotting must not sink the stats
        lines.append(f"\n_(plots skipped: {exc})_")

    report = "\n".join(lines) + "\n"
    _OUT_MD.write_text(report, encoding="utf-8")
    print(report)
    print(f"[analyze] wrote {_OUT_MD.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
