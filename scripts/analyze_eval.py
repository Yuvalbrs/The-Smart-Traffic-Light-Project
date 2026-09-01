"""T-04-02 - Statistical analysis of the eval results, per the frozen pre-registration.

Reads ``data/eval/eval_results.csv`` (T-04-01) and runs the LOCKED analysis plan
(preregistration.md, frozen 2026-06-24, **amended 2026-09-01 A1/A2/A3**): two-sided **Wilcoxon
signed-rank**, paired, with the **Pratt** zero treatment; **Holm-Bonferroni** within three SEPARATE
confirmatory families on the designated test scenario **SCN-05**; median paired difference +
a bootstrap 95% CI as effect size.

Families (SCN-05):
  * C1 / H1 - hybrid vs each of {webster, max_pressure, actuated}, 7 KPIs -> 21 tests.
  * C2 / H2 - hybrid vs plain (the headline forecast ablation), 7 KPIs -> 7 tests.
  * C3 / H3 - hybrid vs random-lstm (information vs capacity), 7 KPIs -> 7 tests.
  * C4 / H1' - plain vs each of {webster, max_pressure, actuated}, 7 KPIs -> 21 tests (**A5**,
    pre-registered before any corrected-world episode existed, so the thesis' fallback headline is
    confirmatory rather than selected after H2 disappointed).

**Amendment A1 (unit of analysis).** One paired observation is ONE EVAL SEED, for every family
alike. A DQN variant's value on an eval seed is the MEAN across its 3 training seeds; a baseline
has one value. Training seeds contribute to the value, never to the count -- the superseded rule
paired over (train_seed x eval_seed) and reused the single baseline number across all 3 train
seeds, so 15 "independent" differences carried only 5 independent baseline observations
(pseudo-replication, p-values biased small, in our favour). Train-seed dispersion is reported as
an ``sd_ts`` column instead of being spent as sample size. Eval seeds are 7000-7014 (n=15): at
n=5 the two-sided signed-rank test has minimum attainable p = 2/2^5 = 0.0625 and cannot reject at
alpha=0.05 for ANY data.

**Amendment A2 (censoring), as corrected by A4.** Gridlock censoring is informative, not missing.
PRIMARY: a censored pair's DIFFERENCE is imputed to strictly exceed every fully-observed |d| in
that comparison, signed by which arm failed - so the pair always takes the top signed rank, no pair
is dropped for censoring, and a mutual failure scores as the tie it is. (A2.2 originally imputed an
extreme VALUE; because the signed-rank test ranks DIFFERENCES, that could give a censored pair the
SMALLEST rank - the opposite of the intent. A4 fixes the scale.)
SENSITIVITY: the same families complete-case (drop the pair if EITHER arm is censored). Both are
reported; if they disagree on a hypothesis, **the disagreement is the result**. The superseded rule
dropped a pair only when BOTH arms were censored, which kept exactly the episodes the treatment won
by not failing -- outcome-dependent selection. A variant's aggregate is censored if ANY of its
training-seed runs is censored (A2.4).

**Amendment A3 (minimum reportable n), as corrected by A4.** A test whose n cannot attain its
family's Holm threshold under any data is reported ``UNDECIDABLE`` and is never reported as a
non-rejection. A3 originally also REDUCED m by the undecidable count; A4 retires that, because n
depends on the data and a data-dependent m weakens the correction exactly when the evidence is
already degraded. m stays pinned at the pre-registered family size.

The 7 KPIs + better-direction are the locked set (preregistration s3). SCN-01..04 are reported as
SUPPORTING/exploratory (raw p, no family correction) - this is where the regime-dependence shows.

Run::

    python -m scripts.analyze_eval                 # -> printed tables + data/eval/analysis.md + plots
"""

from __future__ import annotations

import csv
import math
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
# A2.4: the imputed sentinel sits this far beyond the worst observed value, as a fraction of the
# observed range across both arms (fallback below when the range is 0). Locked in the amendment so
# it cannot be chosen at analysis time.
_SENTINEL_FRAC = 0.10
_SENTINEL_FALLBACK = 1.0


def min_testable_n(m: int, alpha: float = ALPHA) -> int:
    """A1.3/A3: smallest n whose best-case two-sided signed-rank p can clear the Holm threshold.

    With all n paired differences on the same side, the two-sided signed-rank p is 2/2**n -- the
    smallest value the test can ever produce. Holm's most-significant slot in a family of m needs
    p <= alpha/m. So the design is capable of a rejection only when 2/2**n <= alpha/m.

    m=21 (C1) -> n >= 10; m=7 (C2/C3) -> n >= 9; an uncorrected alpha=0.05 -> n >= 6.
    """
    return int(math.ceil(math.log2(2.0 * m / alpha)))


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


def _arm_value(dqn, base, name, is_dqn, scenario, es, kpi):
    """One arm's value on ONE eval seed, per A1.2 + A2.4.

    Returns ``(value, censored, sd_across_train_seeds, present)``.

    A DQN arm averages its uncensored training-seed runs (A1.2) and is censored if ANY of its
    training-seed runs on this eval seed is censored (A2.4) -- averaging a gridlock away behind two
    good seeds would hide exactly what the censor flag exists to expose. A baseline has a single
    run, so no train-seed dispersion exists for it.
    """
    if not is_dqn:
        r = base.get((name, scenario, es))
        if r is None:
            return float("nan"), False, float("nan"), False
        return _num(r, kpi), bool(int(r["gridlock_censored"])), float("nan"), True

    vals: list[float] = []
    censored = False
    present = False
    for ts in TRAIN_SEEDS:
        r = dqn.get((name, ts, scenario, es))
        if r is None:
            continue  # a never-run cell is s8 "missing pair member", handled by the caller
        present = True
        if int(r["gridlock_censored"]):
            censored = True
            continue  # its KPI is computed over completed trips only - not a number to average
        v = _num(r, kpi)
        if not np.isnan(v):
            vals.append(v)
    if not present:
        return float("nan"), False, float("nan"), False
    value = float(np.mean(vals)) if vals else float("nan")
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
    return value, censored, sd, True


def _worst_rank_delta(recs: list[dict]) -> float:
    """A2.2 as CORRECTED by amendment A4: the imputed magnitude, on the DIFFERENCE scale.

    The Wilcoxon signed-rank statistic ranks ``|d_i|``, not the outcome values. The original A2.2
    imputed an extreme VALUE and asserted the resulting pair would take the largest rank - which is
    false whenever the surviving arm happens to sit near the imputed extreme. Measured on a worked
    example the imputed pair ranked 1 of 5, the SMALLEST, so an episode where the comparator
    gridlocked contributed the LEAST evidence. That inverts the intent.

    Imputing on the difference scale makes the guarantee hold by construction: the returned
    magnitude strictly exceeds every fully-observed ``|d|``, so a censored pair always takes the
    top rank, and its SIGN is set by which arm failed (``_series``).
    """
    obs = [abs(r["va"] - r["vb"]) for r in recs
           if not r["ca"] and not r["cb"] and not np.isnan(r["va"]) and not np.isnan(r["vb"])]
    biggest = max(obs, default=0.0)
    if biggest > 0:
        return biggest * (1.0 + _SENTINEL_FRAC)
    # No fully-observed pair to scale against. The former fallback was an ABSOLUTE 1.0, applied
    # identically to num_stops (~0.04) and throughput (~1000) - so every imputed difference became
    # +-1.0, the KPI stopped contributing, and all seven KPIs returned the SAME p-value: the
    # analysis had silently degenerated into a sign test on the censoring pattern, printed as
    # seven per-KPI rows. Fall back to the observed VALUE scale instead, and only to a bare
    # constant when there is no numeric information anywhere (in which case `_series` refuses).
    vals = [r["va"] for r in recs if not r["ca"] and not np.isnan(r["va"])]
    vals += [r["vb"] for r in recs if not r["cb"] and not np.isnan(r["vb"])]
    if len(vals) >= 2 and (spread := max(vals) - min(vals)) > 0:
        return spread * (1.0 + _SENTINEL_FRAC)
    if vals and abs(vals[0]) > 0:
        return abs(vals[0]) * _SENTINEL_FRAC
    return _SENTINEL_FALLBACK


def _series(dqn, base, eval_seeds, scenario, kpi, direction, a_variant, b_name, b_is_dqn, mode):
    """Paired samples for one comparison, one KPI, per A1.2 + A2.2.

    ``mode='primary'``   - censored values take the worst rank; NO pair is dropped for censoring.
    ``mode='complete'``  - the pair is dropped if EITHER arm is censored (the sensitivity analysis).

    Returns ``(a_vals, b_vals, meta)``. Every drop is counted and reported (s8).
    """
    recs: list[dict] = []
    missing = 0
    for es in eval_seeds:
        va, ca, sda, pa = _arm_value(dqn, base, a_variant, True, scenario, es, kpi)
        vb, cb, sdb, pb = _arm_value(dqn, base, b_name, b_is_dqn, scenario, es, kpi)
        if not pa or not pb:
            missing += 1
            continue
        # A genuinely absent number on an UNcensored arm is s8 "missing", not censoring.
        if (not ca and np.isnan(va)) or (not cb and np.isnan(vb)):
            missing += 1
            continue
        recs.append({"es": es, "va": va, "ca": ca, "sda": sda, "vb": vb, "cb": cb, "sdb": sdb})

    cens_a = sum(r["ca"] for r in recs)
    cens_b = sum(r["cb"] for r in recs)
    a_vals: list[float] = []
    b_vals: list[float] = []
    sds: list[float] = []
    dropped_cens = 0
    imputed = 0
    delta = _worst_rank_delta(recs) if mode == "primary" else float("nan")

    for r in recs:
        if mode == "complete":
            if r["ca"] or r["cb"]:
                dropped_cens += 1
                continue
            a_vals.append(r["va"])
            b_vals.append(r["vb"])
        elif r["ca"] and r["cb"]:
            # Mutual failure is a tie: difference exactly 0, Pratt handles the zero (A2.2).
            a_vals.append(0.0)
            b_vals.append(0.0)
            imputed += 2
        elif r["ca"] or r["cb"]:
            # One arm failed. Build the pair so its DIFFERENCE is the extreme one, signed by
            # WHICH arm failed. `_wilcoxon` takes d = a - b; for a lower-is-better KPI a failed
            # A means a > b (d > 0), and the reverse for higher-is-better.
            worse_is_a = bool(r["ca"])
            positive = worse_is_a if direction == "lower" else not worse_is_a
            signed = delta if positive else -delta
            if worse_is_a:
                b_vals.append(r["vb"])
                a_vals.append(r["vb"] + signed)
            else:
                a_vals.append(r["va"])
                b_vals.append(r["va"] - signed)
            imputed += 1
        else:
            a_vals.append(r["va"])
            b_vals.append(r["vb"])
        if not np.isnan(r["sda"]):
            sds.append(r["sda"])

    n_observed = sum(1 for r in recs if not r["ca"] and not r["cb"])
    meta = {
        "missing": missing,
        "dropped_cens": dropped_cens,
        "dropped": missing + dropped_cens,
        "cens_a": cens_a,
        "cens_b": cens_b,
        "imputed": imputed,
        # How many pairs carry REAL measurements on both arms. `_wilcoxon` sees only the numbers
        # and cannot tell an imputed zero from a measured tie: with every pair mutually censored
        # it returned p=1.0, median 0.00 and a ZERO-WIDTH CI, which `build_supporting_regime`
        # published as "no effect detected" on data containing no observations at all. Callers
        # must gate on this, not on len(d).
        "n_observed": n_observed,
        "fully_imputed": bool(recs) and n_observed == 0,
        "n_train_seed_values": int(len(sds)),
        "sd_ts": float(np.mean(sds)) if sds else float("nan"),
    }
    return np.array(a_vals, dtype=float), np.array(b_vals, dtype=float), meta


def _effective_n(a: np.ndarray, b: np.ndarray) -> int:
    """Pairs that can actually carry signed-rank evidence, i.e. non-zero differences.

    A3 compared the RAW pair count against the attainable-n floor. A4.2's both-censored rule
    injects exact zeros, and Pratt zeros raise the attainable minimum p well above 2/2**n - so a
    family could hold 9 mutual failures plus 6 unanimous observed pairs, report n=15, be incapable
    of rejecting under any data, and still occupy a Holm slot taxing its 20 siblings. Measured:
    at n=15 with 6 zeros the best attainable p is 0.0035 against a C1 threshold of 0.00238.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return int(np.count_nonzero(d))


def _wilcoxon(a: np.ndarray, b: np.ndarray):
    """Two-sided Wilcoxon signed-rank (Pratt zeros). Returns (p, median_diff, ci_lo, ci_hi, n)."""
    d = a - b
    n = len(d)
    if np.allclose(d, 0.0) and n >= 1:
        # All differences exactly zero is an unambiguous NON-rejection (p = 1.0), not an
        # untestable hypothesis. Returning NaN here dropped the test out of the Holm
        # family, shrinking m and making every SURVIVING test easier to pass.
        return 1.0, 0.0, 0.0, 0.0, n
    if n < 2:
        return float("nan"), float(np.median(d)) if n else float("nan"), float("nan"), float("nan"), n
    try:
        _, p = wilcoxon(d, zero_method="pratt", alternative="two-sided")
    except ValueError:
        p = float("nan")
    # seed 42: the value recorded as this analysis's provenance in decisions.md and
    # finish-plan.md ("bootstrap 95% CI B=2000 seed=42"). It was 0 here, so anyone
    # reproducing from the recorded provenance got different CI bounds.
    rng = np.random.default_rng(42)
    boots = [np.median(rng.choice(d, size=n, replace=True)) for _ in range(2000)]
    return float(p), float(np.median(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def _holm(pvals: list[float], family_size: int | None = None) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs pass through).

    ``family_size`` pins ``m`` to the PRE-REGISTERED family size (21/7/7) rather than
    letting it shrink to however many tests happened to be computable. Prereg s6 fixes
    those sizes precisely to remove this degree of freedom: a data-dependent m makes the
    correction weaker exactly when censoring has already degraded the evidence. Amendment A3 is
    the ONE sanctioned reduction: a test that cannot reject under any data leaves the family.
    """
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = family_size if family_size is not None else len(idx)
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
    """Run one Holm family over the 7 KPIs, primary + sensitivity, and append both tables.

    Returns the per-test rows so the caller can reuse them for the report.
    """
    prereg_m = len(KPIS) * len(comparisons)
    floor_n = min_testable_n(prereg_m)
    rows = []
    for kpi, klabel, direction in KPIS:
        for b_name, b_is_dqn, blabel in comparisons:
            pa, pb, pm = _series(dqn, base, eval_seeds, scenario, kpi, direction,
                                 a_variant, b_name, b_is_dqn, "primary")
            p_p, med_p, lo_p, hi_p, n_p = _wilcoxon(pa, pb)
            ca, cb, cm = _series(dqn, base, eval_seeds, scenario, kpi, direction,
                                 a_variant, b_name, b_is_dqn, "complete")
            p_c, med_c, lo_c, hi_c, n_c = _wilcoxon(ca, cb)
            # A3 (+A4): a test that cannot clear its family's Holm threshold under ANY data is
            # reported UNDECIDABLE. The count that matters is the EFFECTIVE one - pairs with a
            # non-zero difference - because Pratt zeros cannot contribute signed-rank evidence.
            # A comparison with no fully-observed pair at all is undecidable by definition.
            n_eff = _effective_n(pa, pb)
            undecidable = (n_eff < floor_n) or pm["fully_imputed"]
            # A2.4: an imputed magnitude may not stand as the reported effect size.
            eff = (med_c, lo_c, hi_c, "complete-case") if pm["imputed"] else (med_p, lo_p, hi_p, "primary")
            rows.append({
                "kpi": kpi, "klabel": klabel, "blabel": blabel, "headline": kpi in HEADLINE,
                "p": p_p, "n": n_p, "n_eff": n_eff, "meta": pm, "undecidable": undecidable,
                "med": eff[0], "lo": eff[1], "hi": eff[2], "eff_src": eff[3],
                "p_c": p_c, "n_c": n_c, "meta_c": cm,
            })

    n_und = sum(r["undecidable"] for r in rows)
    # A4 SUPERSEDES A3's m-reduction. A3 shrank m by the undecidable count - but n is a function
    # of the DATA (censoring, missingness), so that is exactly the data-dependent m the _holm
    # docstring warns against, weakening the correction precisely when the evidence is already
    # degraded. Reporting UNDECIDABLE is kept; reducing m is not. Leaving m pinned is the
    # CONSERVATIVE error, which is the acceptable direction.
    m_used = prereg_m
    adj = _holm([float("nan") if r["undecidable"] else r["p"] for r in rows], family_size=m_used)
    adj_c = _holm([float("nan") if r["n_c"] < floor_n else r["p_c"] for r in rows], family_size=m_used)
    for r, a, ac in zip(rows, adj, adj_c):
        r["p_holm"], r["p_holm_c"] = a, ac
        r["sig"] = (not np.isnan(a)) and a < ALPHA
        r["sig_c"] = (not np.isnan(ac)) and ac < ALPHA
        r["agree"] = (r["sig"] == r["sig_c"]) and not r["undecidable"] and r["n_c"] >= floor_n

    m_note = (f"; m = {prereg_m}, PINNED ({n_und} UNDECIDABLE per A3 are reported but do NOT "
              f"shrink m - see A4)" if n_und else f"; m = {prereg_m} as pre-registered")
    lines.append(f"\n### {title}")
    lines.append(f"_n = one value per eval seed (A1.2). Holm family {prereg_m} tests{m_note}. "
                 f"A test needs n >= {floor_n} to be capable of rejection in this family (A1.3)._")
    lines.append("")
    lines.append("| KPI | vs | n | sd_ts | median d (hybrid-other) | 95% CI | eff. src | p_raw | p_holm | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        hl = "**" if r["headline"] else ""
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]" if not np.isnan(r["lo"]) else "n/a"
        sd = f"{r['meta']['sd_ts']:.2f}" if not np.isnan(r["meta"]["sd_ts"]) else "n/a"
        pr = f"{r['p']:.4f}" if not np.isnan(r["p"]) else "n/a"
        if r["undecidable"]:
            why = ("no fully-observed pair" if r["meta"]["fully_imputed"]
                   else f"n_eff={r['n_eff']}<{floor_n}")
            ph, sig = "excluded", f"UNDECIDABLE ({why})"
        else:
            ph = f"{r['p_holm']:.4f}{_stars(r['p_holm'])}" if not np.isnan(r["p_holm"]) else "n/a"
            sig = "**YES**" if r["sig"] else "."
        med = f"{r['med']:+.2f}" if not np.isnan(r["med"]) else "n/a"
        lines.append(f"| {hl}{r['klabel']}{hl} | {r['blabel']} | {r['n']} | {sd} | {med} | "
                     f"{ci} | {r['eff_src']} | {pr} | {ph} | {sig} |")

    tot_imp = sum(r["meta"]["imputed"] for r in rows)
    tot_miss = sum(r["meta"]["missing"] for r in rows)
    clean = ("No episode was censored in this family, so the primary and sensitivity analyses are "
             "numerically identical (A2.3)." if not tot_imp else "")
    lines.append("")
    lines.append(f"_Censoring: {tot_imp} worst-rank imputations across this family; "
                 f"{tot_miss} pair-drops for a missing KPI (s8). {clean}_")

    lines.append("")
    lines.append(f"**Sensitivity (complete-case: pair dropped if EITHER arm censored) - {title}**")
    lines.append("")
    lines.append("| KPI | vs | n_cc | dropped (cens) | p_raw | p_holm | sig | agrees with primary |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        prc = f"{r['p_c']:.4f}" if not np.isnan(r["p_c"]) else "n/a"
        phc = f"{r['p_holm_c']:.4f}" if not np.isnan(r["p_holm_c"]) else "excluded"
        if r["n_c"] < floor_n:
            sigc, agree = f"UNDECIDABLE (n={r['n_c']})", "n/a"
        else:
            sigc = "**YES**" if r["sig_c"] else "."
            agree = "yes" if r["agree"] else "**NO - censoring-dependent**"
        lines.append(f"| {r['klabel']} | {r['blabel']} | {r['n_c']} | {r['meta_c']['dropped_cens']} | "
                     f"{prc} | {phc} | {sigc} | {agree} |")

    disagreements = [r for r in rows if r["n_c"] >= floor_n and not r["undecidable"] and not r["agree"]]
    if disagreements:
        lines.append("")
        lines.append("> **A2.2 disagreement rule fires.** Primary and sensitivity analyses reach "
                     "different conclusions on the tests below. Per the pre-registered rule the "
                     "DISAGREEMENT is the reported result - the conclusion is censoring-dependent, "
                     "and neither analysis may be selected as the headline after the fact.")
        for r in disagreements:
            lines.append(f">   - {r['klabel']} vs {r['blabel']}: primary p_holm="
                         f"{r['p_holm']:.4f} (n={r['n']}), complete-case p_holm="
                         f"{r['p_holm_c']:.4f} (n={r['n_c']}); censored A/B = "
                         f"{r['meta']['cens_a']}/{r['meta']['cens_b']}.")
    return rows


def _supporting_regime(dqn, base, rows, lines):
    """H2 (hybrid vs plain) on the 3 headline KPIs across ALL scenarios - shows regime-dependence."""
    lines.append("\n## Supporting (exploratory, raw p): H2 hybrid-plain across scenarios")
    lines.append("_Where does the forecast help? (no family correction; SCN-05 is the confirmatory one "
                 "above). Exploratory: a non-rejection here is 'not detected at this power', never "
                 "'no effect' (prereg s7)._")
    lines.append("")
    lines.append("| scenario | KPI | median d(hybrid-plain) | n | drop | p_raw |")
    lines.append("|---|---|---|---|---|---|")
    for scenario in sorted({r["scenario"] for r in rows}):
        es = _eval_seeds(rows, scenario)
        for kpi, klabel, direction in KPIS:
            if kpi not in HEADLINE:
                continue
            a, b, meta = _series(dqn, base, es, scenario, kpi, direction, "hybrid", "plain", True, "primary")
            p, med, _lo, _hi, n = _wilcoxon(a, b)
            pr = f"{p:.4f}" if not np.isnan(p) else "n/a"
            md = f"{med:+.2f}" if not np.isnan(med) else "n/a"
            lines.append(f"| {scenario} | {klabel} | {md} | {n} | {meta['dropped']} | {pr} |")


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
    lines = ["# T-04-02 Statistical Analysis (preregistration.md, frozen 2026-06-24, amended 2026-09-01)",
             "",
             "Two-sided Wilcoxon signed-rank, Pratt zero-method, alpha=0.05. Holm-Bonferroni WITHIN each "
             "family. Decision bar = p_holm < 0.05. `**` marks a headline KPI.",
             "",
             "**Amended analysis plan in force (A1/A2/A3, dated 2026-09-01, before any corrected-world "
             "eval episode existed):**",
             "",
             "- **A1** - one paired observation is one EVAL SEED. A DQN value is the mean over its 3 "
             "training seeds; train-seed spread is reported as `sd_ts`, not counted as n. This replaces "
             "the superseded (train_seed x eval_seed) pairing, which reused the single baseline number "
             "across 3 train seeds and so inflated n from 5 to 15 in our favour.",
             "- **A2** - censoring is informative. PRIMARY = worst-rank imputation (no pair dropped for "
             "censoring); SENSITIVITY = complete-case (dropped if EITHER arm censored). Both reported; a "
             "disagreement IS the result. This replaces the superseded both-censored drop, which kept "
             "exactly the episodes the treatment won by not failing.",
             "- **A3+A4** - a test that cannot reject under any data is reported UNDECIDABLE and is "
             "never a non-rejection. A4 RETIRED A3's m-reduction: n is data-dependent, so shrinking m "
             "by the undecidable count weakens the correction exactly when the evidence is already "
             "degraded. m stays PINNED at the pre-registered family size. Decidability uses the "
             "EFFECTIVE n (non-zero differences), since Pratt zeros carry no signed-rank evidence.",
             "- **A4** - censored pairs are imputed on the DIFFERENCE scale (1.10x the largest "
             "fully-observed |d|), signed by which arm failed. A2.2 imputed an extreme VALUE, which "
             "does not control the rank the test actually uses.",
             "- **A5** - family **C4** (`plain` vs the 3 baselines, H1') is confirmatory, pre-registered "
             "before any corrected-world episode existed. Every claim in C1-C4 is CONDITIONAL ON THE "
             "THREE TRAINED CHECKPOINTS (seeds 42/123/2024): the same 3 policies underwrite all "
             "per-eval-seed observations, so the design holds 3 independent draws of the training "
             "process, not 15. Algorithm-level language is not used (A5.4).",
             "",
             f"_Eval seeds present for SCN-05: {len(es05)} (A1.3 locks 15: 7000-7014)._",
             *(["", "> **THIS REPORT IS NOT REPORTABLE.** The input carries "
                 f"{len(es05)} eval seeds; amendment A1.3 requires 15. Below that floor no "
                 "confirmatory test can reject at alpha=0.05 for ANY data, so every result "
                 "here is UNDECIDABLE by construction. Re-run the campaign "
                 "(`scripts.eval_runner`) before citing anything in this file."]
               if len(es05) < 15 else []),
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
    # A5 / H1': the no-forecast agent against the baselines. Corrected within C4 alone - merging
    # it into C1 would change C1's pre-registered m after the fact.
    _family(dqn, base, es05, "SCN-05", "plain",
            [("webster", False, "webster"), ("max_pressure", False, "max_pressure"),
             ("actuated", False, "actuated")],
            "C4 / H1' - plain (no forecast) vs 3 baselines (21 tests, A5)", lines)

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
