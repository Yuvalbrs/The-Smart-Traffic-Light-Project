# `eval_results.csv` here is PRE-FIX and PRE-AMENDMENT. Do not analyse it.

It holds 300 rows from Aug 17-18, generated (a) before the five world-model fixes of
2026-08-28/30 and (b) under the superseded pairing rule. It carries only **5 eval seeds**;
preregistration amendment A1.3 requires **15** (7000-7014).

Consequence, by design: running `python -m scripts.analyze_eval` against this file reports
**every** confirmatory test as `UNDECIDABLE`, because n=5 cannot reject at alpha=0.05 under
any data. That is the amended plan working, not a bug.

A byte-identical copy is in `_archive_pre_gfix_2026-08-30/` under the sha256 manifest.
The amended campaign overwrites this file.
