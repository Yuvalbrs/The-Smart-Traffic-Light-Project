# These tables are PRE-FIX. Do not cite them.

Every file in this directory was generated before the 2026-08-28/30 world-model fixes
(observation clipping, always-firing gradient clip, unenforced max-red bound, reward scale,
lane apportionment) and before the 2026-09-01 preregistration amendments A1-A4.

They are git-tracked, so they survive in place and look current. They are not. A byte-identical
copy sits in `data/eval/_archive_pre_gfix_2026-08-30/analysis/` with the sha256 manifest.

They are kept here only so the repository still builds and the analysis code has something to
read. `python -m scripts.build_analysis` overwrites them once the amended campaign has run.

Specifically invalid in these tables:
- every n (the superseded pairing counted train_seed x eval_seed; A1 counts eval seeds)
- every p-value (biased small by the pseudo-replication A1.1 describes)
- every censoring drop count (the superseded both-censored rule, replaced by A2/A4)
