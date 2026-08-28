"""T-04-02 / T-04-03 - Statistical analysis notebook backend.

Separable, importable analysis+plotting module. T-04-02's DoD requires that the
plotting/table-generation code live outside the notebook, so a plotting break at write-up time has
a recovery path instead of a buried blob. Every number in this module is produced by the LOCKED
statistical core in ``scripts.analyze_eval`` (imported, never reimplemented): ``_pairs``,
``_wilcoxon``, ``_holm``, ``_index``, ``_eval_seeds``, ``_num``, ``_load``, plus the module
constants ``TRAIN_SEEDS``, ``KPIS``, ``HEADLINE``, ``ALPHA``.

Design (per T-04-02 DoD - "the plotting/table-generation code is a separable module"):
    * ``build_*`` functions are pure - they read files but return structured results (pandas
      DataFrames / dicts of DataFrames) with no plotting and no writes.
    * ``render_*`` functions turn a structured result into a markdown string, or draw + save one
      plot. They do no statistics of their own.
    * ``write_table`` is the one place a DataFrame (or dict of DataFrames, for the "one table per
      scenario" outputs T1/T2) is turned into a ``.md`` + ``.csv`` pair on disk.
    * ``main()`` is the only function with meaningful side effects at module scope; importing this
      module does nothing but define functions and constants.

Run::

    python -m scripts.build_analysis          # -> writes every T1-T4, P1-P5, sel/plain into
                                                #    data/eval/analysis/
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_eval import (
    ALPHA,
    HEADLINE,
    KPIS,
    TRAIN_SEEDS,
    _eval_seeds,
    _holm,
    _index,
    _load,
    _num,
    _pairs,
    _stars,
    _wilcoxon,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV = _REPO_ROOT / "data" / "eval" / "eval_results.csv"
_RUNS = _REPO_ROOT / "runs"
_LSTM_DIR = _REPO_ROOT / "checkpoints" / "lstm"
_SEL_PLAIN_LOG = _REPO_ROOT / "runs" / "compare_sel_plain.log"
_OUT_DIR = _REPO_ROOT / "data" / "eval" / "analysis"

TEST_SCENARIO = "SCN-05"
from src.provenance.official import official_lstm_version

DEPLOYED_LSTM_VERSION = official_lstm_version()  # derived, never hand-pinned
HEADLINE_VARIANTS = ("hybrid", "plain", "random-lstm")
BASELINE_ALGOS = ("webster", "max_pressure", "actuated")
# Display order + labels for T1/T2 rows (baselines first, then the 3 DQN variants).
ALGO_LABELS = {
    "webster": "Webster",
    "max_pressure": "Max-pressure",
    "actuated": "SUMO-actuated",
    "hybrid": "DQN-hybrid",
    "plain": "DQN-plain",
    "random-lstm": "DQN-random-lstm",
}
ROW_ORDER = ["webster", "max_pressure", "actuated", "hybrid", "plain", "random-lstm"]

# The 9 headline training runs (3 variants x 3 seeds). Every other runs/* dir (iqn*, *-bc, *-grid,
# *-shift, hybrid-boot*, *_seed7, *_smoke, sanity_*) is a falsified side-experiment and is excluded.
HEADLINE_SEEDS = ("42", "123", "2024")


# --------------------------------------------------------------------------------------------
# Shared data-access helpers (thin wrappers; the stats themselves come from scripts.analyze_eval)
# --------------------------------------------------------------------------------------------


def _algo_key(row: dict) -> str:
    """The variant name for a DQN row, or the algo name for a baseline row."""
    return row["variant"] if row["variant"] else row["algo"]


def load_rows() -> list[dict]:
    """Load ``data/eval/eval_results.csv`` (T-04-01). Thin wrapper around the locked ``_load``."""
    return _load()


def kpi_stats(rows: list[dict], scenario: str, algo: str, kpi_col: str) -> dict:
    """Unpaired descriptive stats for one (scenario, algo, KPI): mean/std over non-NaN values.

    Also reports how many of the rows in this cell are gridlock-censored, so the censoring rate
    is visible next to every mean (preregistration.md s8: a controller cannot "win" a
    throughput/wait comparison by starving inflow without that being shown alongside it).
    """
    sub = [r for r in rows if r["scenario"] == scenario and _algo_key(r) == algo]
    vals = [_num(r, kpi_col) for r in sub]
    valid = [v for v in vals if not np.isnan(v)]
    n_censored = sum(1 for r in sub if int(r["gridlock_censored"]))
    return {
        "mean": float(np.mean(valid)) if valid else float("nan"),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "n_valid": len(valid),
        "n_total": len(sub),
        "n_censored": n_censored,
    }


def _fmt_cell(stats: dict) -> str:
    if stats["n_valid"] == 0:
        return "n/a (all censored)"
    return f"{stats['mean']:.2f} ± {stats['std']:.2f}"


# --------------------------------------------------------------------------------------------
# Confirmatory Holm-Bonferroni families (C1/C2/C3, SCN-05 only) - reuses _pairs/_wilcoxon/_holm
# unmodified. This is the single source of truth for every "significant yes/no" claim in the
# report; T1/T2 annotate their SCN-05 columns from these DataFrames rather than recomputing.
# --------------------------------------------------------------------------------------------


def build_confirmatory_family(
    rows: list[dict],
    a_variant: str,
    comparisons: list[tuple[str, bool, str]],
    scenario: str = TEST_SCENARIO,
) -> pd.DataFrame:
    """One Holm family: ``a_variant`` vs each of ``comparisons`` on the 7 KPIs.

    ``comparisons`` is a list of ``(b_name, b_is_dqn, label)``, matching ``scripts.analyze_eval._pairs``.
    Returns one row per (KPI, comparison) with the raw p, Holm-adjusted p, and the effect size.
    """
    dqn, base = _index(rows)
    es = _eval_seeds(rows, scenario)
    records = []
    for kpi_col, kpi_label, direction in KPIS:
        for b_name, b_is_dqn, blabel in comparisons:
            a, b, dropped = _pairs(dqn, base, es, scenario, kpi_col, a_variant, b_name, b_is_dqn)
            p, med, lo, hi, n = _wilcoxon(a, b)
            records.append(
                {
                    "kpi": kpi_label,
                    "kpi_col": kpi_col,
                    "direction": direction,
                    "headline": kpi_col in HEADLINE,
                    "comparison": blabel,
                    "n": n,
                    "dropped": dropped,
                    "median_diff": med,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_raw": p,
                }
            )
    df = pd.DataFrame.from_records(records)
    # Holm divides by the PRE-REGISTERED family size (21/7/7), not by however many tests
    # survived censoring - a data-dependent m weakens the correction exactly when the
    # evidence is already degraded (prereg s6).
    df["p_holm"] = _holm(df["p_raw"].tolist(), family_size=len(df))
    df["stars"] = df["p_holm"].apply(_stars)
    df["significant"] = df["p_holm"].apply(lambda p: bool(p < ALPHA) if not np.isnan(p) else False)
    df["family_size"] = int(len(df))  # pre-registered size
    df["family_testable"] = int(df["p_raw"].notna().sum())  # how many were computable
    return df


def build_family_c1(rows: list[dict]) -> pd.DataFrame:
    """C1/H1: hybrid vs {webster, max_pressure, actuated} x 7 KPIs = 21 hypotheses (SCN-05)."""
    return build_confirmatory_family(
        rows,
        "hybrid",
        [("webster", False, "Webster"), ("max_pressure", False, "Max-pressure"), ("actuated", False, "SUMO-actuated")],
    )


def build_family_c2(rows: list[dict]) -> pd.DataFrame:
    """C2/H2: hybrid vs plain (headline forecast ablation) x 7 KPIs = 7 hypotheses (SCN-05)."""
    return build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")])


def build_family_c3(rows: list[dict]) -> pd.DataFrame:
    """C3/H3: hybrid vs random-lstm (information vs capacity) x 7 KPIs = 7 hypotheses (SCN-05)."""
    return build_confirmatory_family(rows, "hybrid", [("random-lstm", True, "DQN-random-lstm")])


def build_supporting_regime(rows: list[dict]) -> pd.DataFrame:
    """Exploratory: hybrid vs plain and hybrid vs random-lstm on the 3 headline KPIs, ALL 5
    scenarios, raw p, no family correction. Shows where the forecast helps / hurts by regime -
    this is where the SCN-01 hybrid-worse-than-plain finding lives.
    """
    dqn, base = _index(rows)
    records = []
    scenarios = sorted({r["scenario"] for r in rows})
    for scenario in scenarios:
        es = _eval_seeds(rows, scenario)
        for b_name, blabel in [("plain", "DQN-plain"), ("random-lstm", "DQN-random-lstm")]:
            for kpi_col, kpi_label, direction in KPIS:
                if kpi_col not in HEADLINE:
                    continue
                a, b, dropped = _pairs(dqn, base, es, scenario, kpi_col, "hybrid", b_name, True)
                p, med, lo, hi, n = _wilcoxon(a, b)
                verdict = "no effect detected at n=15" if (np.isnan(p) or p >= ALPHA) else (
                    "hybrid better" if (med < 0) == (direction == "lower") else "hybrid worse"
                )
                records.append(
                    {
                        "scenario": scenario,
                        "vs": blabel,
                        "kpi": kpi_label,
                        "direction": direction,
                        "median_diff_hybrid_minus_other": med,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n": n,
                        "dropped": dropped,
                        "p_raw": p,
                        "verdict": verdict,
                    }
                )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# Honest findings that must survive - computed, not hand-typed, so the notebook prose can quote
# these numbers directly instead of risking a transcription error.
# --------------------------------------------------------------------------------------------


def build_honest_findings(rows: list[dict]) -> pd.DataFrame:
    """Descriptive stats with gridlock-censored rows FULLY excluded (stricter than T1/T2's
    non-NaN filter, which keeps censored-but-valid rows). This is the filter the SCN-01
    hybrid-worse-than-plain finding uses: "among non-censored episodes". T1/T2 additionally
    report a per-cell censoring rate so the two views (all non-NaN data vs non-censored-only)
    are both auditable; this table isolates the latter for the specific honest-finding claims.
    """
    records = []
    for scenario in ("SCN-01", "SCN-04"):  # SCN-01: hybrid worse: SCN-04: DQN beats Webster
        for algo in ("hybrid", "plain", "webster"):
            for kpi_col in ("avg_waiting_time", "throughput", "worst_movement_max_wait"):
                sub = [r for r in rows if r["scenario"] == scenario and _algo_key(r) == algo
                       and not int(r["gridlock_censored"])]
                vals = [_num(r, kpi_col) for r in sub]
                vals = [v for v in vals if not np.isnan(v)]
                records.append(
                    {
                        "scenario": scenario,
                        "algorithm": ALGO_LABELS[algo],
                        "kpi": kpi_col,
                        "n_noncensored": len(vals),
                        "mean_noncensored": float(np.mean(vals)) if vals else float("nan"),
                        "std_noncensored": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    }
                )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# T1 - Main results (one table per scenario; rows = all algorithms; cols = 7 KPIs)
# --------------------------------------------------------------------------------------------


def _sig_marker(rows: list[dict], scenario: str, baseline_or_variant: str, kpi_col: str,
                 family_lookup: dict) -> str:
    """Significance annotation for one (scenario, comparator, KPI) cell, vs hybrid.

    On SCN-05 (the designated test scenario) this looks up the Holm-adjusted p from the relevant
    confirmatory family (C1 for baselines, C2/C3 for plain/random-lstm) - the only p-values this
    project is allowed to call "significant". On every other (supporting/exploratory) scenario it
    computes a RAW, uncorrected p and marks it distinctly, per preregistration.md s6/s7.
    """
    # Bracket notation, not literal `*`/`**` glyphs: a bare `**` inside a markdown table cell is
    # unmatched bold-emphasis syntax and renders unpredictably across markdown engines.
    if scenario == TEST_SCENARIO and (baseline_or_variant, kpi_col) in family_lookup:
        p_holm = family_lookup[(baseline_or_variant, kpi_col)]
        if np.isnan(p_holm) or p_holm >= ALPHA:
            return ""
        return f"[p<0.01,holm]" if p_holm < 0.01 else f"[p<0.05,holm]"
    # supporting/exploratory: raw p, own scenario, not part of any Holm family
    dqn, base = _index(rows)
    es = _eval_seeds(rows, scenario)
    is_dqn = baseline_or_variant in HEADLINE_VARIANTS and baseline_or_variant != "hybrid"
    a, b, _dropped = _pairs(dqn, base, es, scenario, kpi_col, "hybrid", baseline_or_variant, is_dqn)
    p, _med, _lo, _hi, _n = _wilcoxon(a, b)
    if np.isnan(p) or p >= ALPHA:
        return ""
    return f"[p<0.01,raw]" if p < 0.01 else f"[p<0.05,raw]"


def build_t1_main_results(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """T1: one DataFrame per scenario. Rows = all 6 algorithms. Cols = 7 KPI cells (mean +/- std,
    with a significance marker on baseline rows showing whether DQN-hybrid differs from that
    baseline for that KPI/scenario) plus n_valid/n_censored bookkeeping columns.

    Marker convention (documented here because a table cell can't hold a footnote): the marker on
    a BASELINE row is for "hybrid vs this baseline" (H1). DQN-variant rows carry no marker in T1 -
    their pairwise comparisons against hybrid are T2's job. `(holm)` = Holm-adjusted p from the C1
    confirmatory family (SCN-05 only, the only scenario allowed a corrected significance claim).
    `(raw)` = uncorrected raw p on a supporting/exploratory scenario - never a confirmatory claim.
    """
    c1 = build_family_c1(rows)
    # normalize comparison labels back to algo keys used in eval_results.csv
    label_to_algo = {"Webster": "webster", "Max-pressure": "max_pressure", "SUMO-actuated": "actuated"}
    holm_lookup = {(label_to_algo[row["comparison"]], row["kpi_col"]): row["p_holm"] for _, row in c1.iterrows()}

    scenarios = sorted({r["scenario"] for r in rows})
    tables: dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        records = []
        for algo in ROW_ORDER:
            rec = {"algorithm": ALGO_LABELS[algo]}
            for kpi_col, kpi_label, _direction in KPIS:
                s = kpi_stats(rows, scenario, algo, kpi_col)
                cell = _fmt_cell(s)
                if algo in BASELINE_ALGOS and algo != "hybrid":
                    marker = _sig_marker(rows, scenario, algo, kpi_col, holm_lookup)
                    if marker:
                        cell = f"{cell} {marker}"
                rec[kpi_label] = cell
            rec["n_censored/n_total"] = (
                f"{sum(1 for r in rows if r['scenario'] == scenario and _algo_key(r) == algo and int(r['gridlock_censored']))}"
                f"/{sum(1 for r in rows if r['scenario'] == scenario and _algo_key(r) == algo)}"
            )
            records.append(rec)
        tables[scenario] = pd.DataFrame.from_records(records)
    return tables


# --------------------------------------------------------------------------------------------
# T2 - Ablation results (rows = DQN variants only)
# --------------------------------------------------------------------------------------------


def build_t2_ablation_results(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """T2: same format as T1, rows restricted to the 3 DQN variants. `plain` and `random-lstm`
    rows carry a marker for "vs hybrid" (Holm-adjusted from C2/C3 on SCN-05, raw elsewhere).

    The switch-penalty lambda sweep {0, 0.05, 0.1, 0.5} that T-04-03's ablation matrix calls for
    was never run - there is no data for it. That gap is stated in the caption, not simulated.
    """
    c2 = build_family_c2(rows)
    c3 = build_family_c3(rows)
    holm_lookup = {("plain", row["kpi_col"]): row["p_holm"] for _, row in c2.iterrows()}
    holm_lookup.update({("random-lstm", row["kpi_col"]): row["p_holm"] for _, row in c3.iterrows()})

    scenarios = sorted({r["scenario"] for r in rows})
    tables: dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        records = []
        for algo in HEADLINE_VARIANTS:
            rec = {"algorithm": ALGO_LABELS[algo]}
            for kpi_col, kpi_label, _direction in KPIS:
                s = kpi_stats(rows, scenario, algo, kpi_col)
                cell = _fmt_cell(s)
                if algo != "hybrid":
                    marker = _sig_marker(rows, scenario, algo, kpi_col, holm_lookup)
                    if marker:
                        cell = f"{cell} {marker}"
                rec[kpi_label] = cell
            records.append(rec)
        tables[scenario] = pd.DataFrame.from_records(records)
    return tables


# --------------------------------------------------------------------------------------------
# T3 - LSTM standalone eval
# --------------------------------------------------------------------------------------------


def build_t3_lstm_standalone() -> pd.DataFrame:
    """T3: all 6 LSTM training attempts from checkpoints/lstm/*__report.json.

    The report jsons record aggregate val/test MSE over the LSTM's own held-out data split
    (data_version=data-8eb28eecdefb); they are NOT decomposed per-scenario, so there is no
    SCN-05-only forecast MSE in the source data - the aggregate val_mse/test_mse is reported as
    the closest available standalone-eval metric, and that gap is stated explicitly here rather
    than fabricating a per-scenario number.
    """
    import json

    records = []
    for path in sorted(_LSTM_DIR.glob("*__report.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        gate = d.get("gate", {})
        ss_near = gate.get("ss_h1", gate.get("ss_near", float("nan")))
        ss_far = gate.get("ss_h3", gate.get("ss_far", float("nan")))
        records.append(
            {
                "lstm_version": d.get("lstm_version"),
                "seed": d.get("seed"),
                "val_mse": d.get("val_mse"),
                "test_mse": d.get("test_mse"),
                "r2_val": d.get("r2_val"),
                "skill_score_nearest_horizon": ss_near,
                "skill_score_farthest_horizon": ss_far,
                "gate_verdict": gate.get("verdict"),
                "ship": gate.get("ship"),
                "deployed": d.get("lstm_version") == DEPLOYED_LSTM_VERSION,
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# T4 - Wall-clock budget
# --------------------------------------------------------------------------------------------


def build_t4_wallclock_budget() -> pd.DataFrame:
    """T4: training/eval time per condition. No timing field was found anywhere in the run
    artifacts for the 9 headline runs (config.yaml, episodes.csv, steps.csv, validation.csv,
    matrix_summary.json, matrix.log all inspected) - this is stated as a gap, not estimated.
    """
    import yaml

    records = []
    for variant in HEADLINE_VARIANTS:
        for seed in HEADLINE_SEEDS:
            run_dir = _RUNS / f"{variant}_seed{seed}"
            cfg_path = run_dir / "config.yaml"
            ep_path = run_dir / "episodes.csv"
            n_episodes = n_steps = None
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                n_episodes = cfg.get("n_episodes")
            if ep_path.exists():
                ep_rows = list(csv.DictReader(ep_path.open(encoding="utf-8")))
                n_steps = int(ep_rows[-1]["total_steps"]) if ep_rows else None
            records.append(
                {
                    "condition": f"{variant}_seed{seed}",
                    "episodes_completed": n_episodes,
                    "total_env_steps": n_steps,
                    "training_wall_clock_s": "NOT RECORDED",
                    "eval_wall_clock_s": "NOT RECORDED",
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# sel/plain headline table (the shipped product) - parsed from runs/compare_sel_plain.log so the
# numbers are never hand-copied. Follows scripts/compare_selector_plain.py's table shape.
# --------------------------------------------------------------------------------------------

_SEL_SCENARIO_RE = re.compile(r"^===\s*(SCN-\d+)\s*===")
_SEL_ROW_RE = re.compile(
    r"^\s*(\S+)\s+wait=\s*(nan|-?[\d.]+)\s*\|\s*thru=\s*(nan|-?[\d.]+)\s*\|\s*grid=\s*(-?\d+)%"
)


def build_sel_plain_table(log_path: Path = _SEL_PLAIN_LOG) -> pd.DataFrame:
    """Parse every scenario x condition row out of runs/compare_sel_plain.log (webster + plain vs
    sel/plain for each of the 3 seeds), including SCN-06 - required, never filtered out.
    """
    text = log_path.read_text(encoding="utf-8")
    records = []
    scenario = None
    for line in text.splitlines():
        m_scn = _SEL_SCENARIO_RE.match(line)
        if m_scn:
            scenario = m_scn.group(1)
            continue
        m_row = _SEL_ROW_RE.match(line)
        if m_row and scenario is not None:
            name, wait, thru, grid = m_row.groups()
            records.append(
                {
                    "scenario": scenario,
                    "condition": name,
                    "avg_wait_s": float("nan") if wait == "nan" else float(wait),
                    "throughput": float("nan") if thru == "nan" else float(thru),
                    "pct_gridlock": float(grid),
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# Webster feasibility (locked N/A rule check) - does a Webster-N/A / FixedTime-120 case exist?
# --------------------------------------------------------------------------------------------


def build_webster_feasibility() -> pd.DataFrame:
    """Y (flow ratio) and Webster status for every scenario present in eval_results.csv.

    preregistration.md s8: Y >= 1.00 -> Webster N/A, named FixedTime-120 fallback substituted,
    never blanked. This checks whether that case actually occurs in the data (it does not
    fabricate one if it doesn't).
    """
    from src.baselines.webster import webster_plan_for_scenario
    from src.scenarios.config import SCENARIO_DIR, load_scenario

    rows = load_rows()
    scenarios = sorted({r["scenario"] for r in rows})
    records = []
    for scenario in scenarios:
        num = scenario.split("-")[1]
        scn = load_scenario(SCENARIO_DIR / f"scn_{num}.yaml")
        plan = webster_plan_for_scenario(scn)
        records.append(
            {
                "scenario": scenario,
                "flow_ratio_Y": plan.flow_ratio_Y,
                "status": plan.status,
                "cycle_s": plan.cycle_s,
                "webster_na": plan.status == "na",
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------------
# P1 - Reward curves (real: ep_reward from runs/{variant}_seed{42,123,2024}/episodes.csv)
# --------------------------------------------------------------------------------------------

P1_ROLLING_WINDOW = 10


def build_p1_data(window: int = P1_ROLLING_WINDOW) -> dict[str, dict[str, np.ndarray]]:
    """Per-variant: rolling-mean ep_reward per seed, then mean/std across the 3 seeds per episode."""
    out = {}
    for variant in HEADLINE_VARIANTS:
        per_seed = []
        for seed in HEADLINE_SEEDS:
            ep_path = _RUNS / f"{variant}_seed{seed}" / "episodes.csv"
            ep_rows = list(csv.DictReader(ep_path.open(encoding="utf-8")))
            rewards = np.array([float(r["ep_reward"]) for r in ep_rows])
            smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().to_numpy()
            per_seed.append(smoothed)
        arr = np.vstack(per_seed)  # (3 seeds, n_episodes)
        out[variant] = {
            "episode": np.arange(arr.shape[1]),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0, ddof=0),
        }
    return out


def render_p1(data: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"hybrid": "tab:blue", "plain": "tab:orange", "random-lstm": "tab:green"}
    for variant, d in data.items():
        c = colors.get(variant)
        ax.plot(d["episode"], d["mean"], label=ALGO_LABELS[variant], color=c)
        ax.fill_between(d["episode"], d["mean"] - d["std"], d["mean"] + d["std"], color=c, alpha=0.2)
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"episode reward ({P1_ROLLING_WINDOW}-episode rolling mean)")
    ax.set_title(
        f"P1: training reward curves, plain/hybrid/random-lstm, seeds {{{', '.join(HEADLINE_SEEDS)}}}\n"
        f"shaded band = ±1 std across the 3 seeds; smoothed with a {P1_ROLLING_WINDOW}-episode rolling mean"
    )
    ax.legend()
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# P2 - Validation reward over episodes (real: runs/*/validation.csv, every 25 episodes)
# --------------------------------------------------------------------------------------------


def build_p2_data() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for variant in HEADLINE_VARIANTS:
        per_seed_means = []
        episodes_ref = None
        for seed in HEADLINE_SEEDS:
            val_path = _RUNS / f"{variant}_seed{seed}" / "validation.csv"
            val_rows = list(csv.DictReader(val_path.open(encoding="utf-8")))
            episodes = np.array([int(r["episode"]) for r in val_rows])
            means = np.array([float(r["val_mean_reward"]) for r in val_rows])
            if episodes_ref is None:
                episodes_ref = episodes
            per_seed_means.append(means)
        arr = np.vstack(per_seed_means)
        out[variant] = {"episode": episodes_ref, "mean": arr.mean(axis=0), "std": arr.std(axis=0, ddof=0)}
    return out


def render_p2(data: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"hybrid": "tab:blue", "plain": "tab:orange", "random-lstm": "tab:green"}
    for variant, d in data.items():
        c = colors.get(variant)
        ax.plot(d["episode"], d["mean"], marker="o", label=ALGO_LABELS[variant], color=c)
        ax.fill_between(d["episode"], d["mean"] - d["std"], d["mean"] + d["std"], color=c, alpha=0.2)
    ax.set_xlabel("training episode")
    ax.set_ylabel("validation mean reward (val_mean_reward, SCN-04)")
    ax.set_title(
        "P2: validation reward over training, evaluated every 25 episodes (5 val episodes/point)\n"
        f"shaded band = ±1 std across seeds {{{', '.join(HEADLINE_SEEDS)}}}"
    )
    ax.legend()
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# P3 - Per-movement max wait bar chart. PROXY: the SCN-05_*.jsonl traces record per-vehicle
# {lane, movement_id, speed, x, y} per 1 Hz frame but no per-movement WAIT aggregate, so the real
# 12-bar-per-controller chart cannot be built from what's on disk without re-deriving waits from
# raw trajectories (out of scope here - this module reads existing data, it does not recompute
# KPIs from traces). Falls back to worst_movement_max_wait per controller from eval_results.csv,
# one bar per controller, clearly labeled PROXY.
# --------------------------------------------------------------------------------------------


def build_p3_data(rows: list[dict], scenario: str = TEST_SCENARIO) -> dict[str, dict]:
    out = {}
    for algo in ROW_ORDER:
        s = kpi_stats(rows, scenario, algo, "worst_movement_max_wait")
        out[algo] = s
    return out


def render_p3(data: dict, out_path: Path, scenario: str = TEST_SCENARIO) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import textwrap

    fig, ax = plt.subplots(figsize=(9, 6.5))
    algos = [a for a in ROW_ORDER if data[a]["n_valid"] > 0]
    means = [data[a]["mean"] for a in algos]
    stds = [data[a]["std"] for a in algos]
    labels = [ALGO_LABELS[a] for a in algos]
    ax.bar(labels, means, yerr=stds, capsize=4, color="tab:red", alpha=0.75)
    ax.set_ylabel("worst-movement max wait (s)")
    ax.set_title(f"P3 [PROXY]: worst-movement max wait per controller, {scenario}", fontsize=12)
    caption = (
        "PROXY for the real 12-bars-x-N-algorithms per-movement chart: the SCN-05 JSONL traces "
        "record per-vehicle {lane, movement_id, speed, x, y} at 1 Hz but no per-movement wait "
        "aggregate, so the real chart would need re-deriving per-movement waiting time from raw "
        "trajectories (out of scope for this analysis pass). This proxy shows only the single "
        "worst movement per controller (5b in eval_results.csv), not all 12."
    )
    wrapped = "\n".join(textwrap.wrap(caption, width=95))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8.5)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# P4 - KPI distributions box plot (real: eval_results.csv, SCN-05, n=15 for DQN / n=5 for
# baselines - baselines are NOT replicated to 15; the true sample size is shown per box)
# --------------------------------------------------------------------------------------------


def build_p4_data(rows: list[dict], scenario: str = TEST_SCENARIO) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for kpi_col, kpi_label, _direction in KPIS:
        out[kpi_label] = {}
        for algo in ROW_ORDER:
            sub = [r for r in rows if r["scenario"] == scenario and _algo_key(r) == algo]
            vals = [_num(r, kpi_col) for r in sub]
            out[kpi_label][algo] = [v for v in vals if not np.isnan(v)]
    return out


def render_p4(data: dict, out_path: Path, scenario: str = TEST_SCENARIO) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kpi_labels = list(data.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, kpi_label in enumerate(kpi_labels):
        ax = axes[i]
        algos = [a for a in ROW_ORDER if data[kpi_label][a]]
        box_data = [data[kpi_label][a] for a in algos]
        labels = [f"{ALGO_LABELS[a]}\n(n={len(data[kpi_label][a])})" for a in algos]
        ax.boxplot(box_data, tick_labels=labels)
        ax.set_title(kpi_label, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
    for j in range(len(kpi_labels), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"P4: KPI distributions by controller, {scenario} (the designated test scenario)\n"
        "one box per algorithm per KPI; sample size shown per box (DQN variants n=15, baselines n=5 - not replicated)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# P5 - Forecast accuracy during RL training (real: ss_rolling from runs/*/episodes.csv, hybrid +
# random-lstm only; `plain` has forecast=false so its ss_rolling column is entirely empty - excluded)
# --------------------------------------------------------------------------------------------

P5_ROLLING_WINDOW = 20


def build_p5_data(window: int = P5_ROLLING_WINDOW) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    grid = np.arange(300)
    for variant in ("hybrid", "random-lstm"):
        per_seed_interp = []
        for seed in HEADLINE_SEEDS:
            ep_path = _RUNS / f"{variant}_seed{seed}" / "episodes.csv"
            ep_rows = list(csv.DictReader(ep_path.open(encoding="utf-8")))
            eps, vals = [], []
            for r in ep_rows:
                if r["ss_rolling"] not in ("", "nan"):
                    eps.append(int(r["episode"]))
                    vals.append(float(r["ss_rolling"]))
            eps = np.array(eps)
            vals = pd.Series(vals).rolling(window, min_periods=1).mean().to_numpy()
            interp = np.interp(grid, eps, vals, left=np.nan, right=np.nan)
            per_seed_interp.append(interp)
        arr = np.vstack(per_seed_interp)
        # Grid points before the first / after the last non-NaN episode for a given seed are NaN by
        # design (np.interp left=right=nan); an all-NaN column at the grid edges is expected, not
        # an error - suppress numpy's "empty slice" warning for exactly that case.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
        out[variant] = {"episode": grid, "mean": mean, "std": std}
    return out


def render_p5(data: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"hybrid": "tab:blue", "random-lstm": "tab:green"}
    for variant, d in data.items():
        c = colors.get(variant)
        ax.plot(d["episode"], d["mean"], label=ALGO_LABELS[variant], color=c)
        ax.fill_between(d["episode"], d["mean"] - d["std"], d["mean"] + d["std"], color=c, alpha=0.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="skill score = 0 (persistence baseline)")
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"rolling skill score (ss_rolling, {P5_ROLLING_WINDOW}-episode rolling mean)")
    ax.set_title(
        "P5: in-RL forecast skill during training (hybrid, random-lstm only - plain has forecast=false,\n"
        "its ss_rolling column is entirely empty and is excluded). Documents distribution-shift: the LSTM was\n"
        "trained offline and its skill score visibly drifts/degrades as the RL policy shifts the state distribution."
    )
    ax.legend()
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Markdown rendering + file writing (no statistics happens here)
# --------------------------------------------------------------------------------------------


def _md_escape(value: str) -> str:
    """Escape markdown special chars (`|`, `*`) so a cell's content can't be misread as table
    structure or unmatched bold-emphasis syntax by a markdown renderer."""
    return value.replace("|", "\\|").replace("*", "\\*")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """A dependency-free markdown table renderer (tabulate is not installed in this env)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_md_escape(str(row[c])) for c in cols) + " |")
    return "\n".join(lines)


def write_table(result, name: str, out_dir: Path, title: str, caption: str = "") -> None:
    """Write one table (or dict-of-scenario-tables) as BOTH ``<name>.md`` and ``<name>.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_lines = [f"# {title}", ""]
    if caption:
        md_lines += [caption, ""]

    if isinstance(result, dict):
        csv_frames = []
        for scenario, df in result.items():
            tag = " (confirmatory - SCN-05)" if scenario == TEST_SCENARIO else " (supporting/exploratory)"
            md_lines += [f"## {scenario}{tag}", "", _df_to_markdown(df), ""]
            df2 = df.copy()
            df2.insert(0, "scenario", scenario)
            csv_frames.append(df2)
        pd.concat(csv_frames, ignore_index=True).to_csv(out_dir / f"{name}.csv", index=False)
    else:
        md_lines += [_df_to_markdown(result), ""]
        result.to_csv(out_dir / f"{name}.csv", index=False)

    (out_dir / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# main() - the only function with real side effects
# --------------------------------------------------------------------------------------------


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console safety
    except (AttributeError, ValueError):
        pass

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    # Confirmatory families (also feed T1/T2 markers)
    c1, c2, c3 = build_family_c1(rows), build_family_c2(rows), build_family_c3(rows)
    write_table(c1, "family_C1_hybrid_vs_baselines", _OUT_DIR,
                "C1/H1 - DQN-hybrid vs 3 baselines, SCN-05 (21 hypotheses, Holm-Bonferroni)")
    write_table(c2, "family_C2_hybrid_vs_plain", _OUT_DIR,
                "C2/H2 - DQN-hybrid vs DQN-plain, SCN-05 (7 hypotheses, Holm-Bonferroni)")
    write_table(c3, "family_C3_hybrid_vs_random_lstm", _OUT_DIR,
                "C3/H3 - DQN-hybrid vs DQN-random-lstm, SCN-05 (7 hypotheses, Holm-Bonferroni)")
    write_table(build_supporting_regime(rows), "supporting_regime_hybrid_vs_ablations", _OUT_DIR,
                "Supporting/exploratory: hybrid vs plain / random-lstm, headline KPIs, all 5 scenarios, raw p")
    write_table(build_honest_findings(rows), "honest_findings_noncensored", _OUT_DIR,
                "Honest findings, gridlock-censored episodes FULLY excluded (stricter than T1/T2)",
                "SCN-01: DQN-hybrid is worse than DQN-plain on avg wait among non-censored episodes. "
                "SCN-04: the DQN-beats-Webster story. Both are reported here, not softened.")

    # T1 / T2
    write_table(
        build_t1_main_results(rows), "T1_main_results", _OUT_DIR,
        "T1: Main results - one table per scenario, all algorithms, 7 KPIs (mean +/- std)",
        "Marker on a baseline row = hybrid vs that baseline. `(holm)` on SCN-05 = Holm-adjusted p "
        "from the C1 confirmatory family (the only corrected significance claim in this project). "
        "`(raw)` elsewhere = uncorrected raw p, supporting/exploratory only, never a confirmatory claim.",
    )
    write_table(
        build_t2_ablation_results(rows), "T2_ablation_results", _OUT_DIR,
        "T2: Ablation results - DQN variants only, 7 KPIs (mean +/- std)",
        "Marker on plain/random-lstm rows = vs hybrid ((holm) on SCN-05 from C2/C3, (raw) elsewhere). "
        "The switch-penalty lambda sweep {0, 0.05, 0.1, 0.5} required by T-04-03's ablation matrix "
        "was NEVER RUN - there is no data for it in this repo; it is not simulated here.",
    )

    # T3 / T4
    write_table(build_t3_lstm_standalone(), "T3_lstm_standalone", _OUT_DIR,
                "T3: LSTM standalone eval - all 6 training attempts",
                f"Deployed = {DEPLOYED_LSTM_VERSION} (SHIP_WITH_CAVEAT). Only 2 of 6 attempts passed "
                "the skill-score gate. MSE is the LSTM's own held-out val/test split, not decomposed "
                "per-scenario in the source jsons - no SCN-05-only forecast MSE exists on disk.")
    write_table(build_t4_wallclock_budget(), "T4_wallclock_budget", _OUT_DIR,
                "T4: Wall-clock budget per condition",
                "GAP: no wall-clock timing field was found in config.yaml, episodes.csv, steps.csv, "
                "validation.csv, matrix_summary.json, or matrix.log for any of the 9 headline runs. "
                "This is reported as a gap, not estimated.")

    # sel/plain headline table
    write_table(build_sel_plain_table(), "sel_plain_headline", _OUT_DIR,
                "sel/plain: EpisodeLevelSelector(plain-DQN, Webster) vs plain DQN, per scenario x seed",
                "Parsed verbatim from runs/compare_sel_plain.log (scripts/compare_selector_plain.py). "
                "Every row present in the log is included, SCN-06 included.")

    # Webster feasibility
    write_table(build_webster_feasibility(), "webster_feasibility", _OUT_DIR,
                "Webster feasibility (flow ratio Y) per scenario in eval_results.csv",
                "preregistration.md s8: Y >= 1.00 -> Webster N/A, FixedTime-120 substituted. "
                "Checked against the actual scenario definitions (not fabricated).")

    # Plots
    render_p1(build_p1_data(), _OUT_DIR / "P1_reward_curves.png")
    render_p2(build_p2_data(), _OUT_DIR / "P2_validation_reward.png")
    render_p3(build_p3_data(rows), _OUT_DIR / "P3_per_movement_max_wait_PROXY.png")
    render_p4(build_p4_data(rows), _OUT_DIR / "P4_kpi_distributions.png")
    render_p5(build_p5_data(), _OUT_DIR / "P5_forecast_accuracy_training.png")

    print(f"[build_analysis] wrote all T1-T4 / P1-P5 / sel-plain / family tables to "
          f"{_OUT_DIR.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
