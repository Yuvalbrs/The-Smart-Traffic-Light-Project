"""The ONE place that names the deployed forecasters — one per DQN training seed.

The checkpoint was pinned by hand in five files (``eval_runner``, ``train_matrix``,
``build_analysis``, ``bootstrap_forecaster``, ``collect_dqn_traces``). When the
forecaster was retrained after the gridlock-artefact fixes, two were updated and three
were not - so the analysis layer reported the provenance of a forecaster the experiment
had not used, and the final deliverable printed a stale version string next to correct
results. Sync enforced by a comment is not sync (decisions.md 2026-08-28).

**Amendment A6.4 (2026-09-01) made the pin per-seed.** The retrain rule gates on the MEAN
skill score over seeds {42, 123, 2024} but never said which of the three checkpoints ships -
and on the pre-split runs the canonical seed 42 failed its own gate while 123 and 2024 passed,
so "deploy 42", "deploy the best" and "deploy the median" were three different experiments,
still selectable after the numbers were visible. Pairing each DQN training seed to the
forecaster of the same seed removes the choice entirely: there is nothing left to select.

It also removes a hidden pseudo-replication. Under the single pin, all three "independent"
hybrid training runs carried the SAME forecaster, so forecaster-training variance was
replicated zero times - while the gate verdict is known to flip sign across forecaster seeds
on identical data and identical hyperparameters. This does not lift A5.4's caveat (three
draws are still three), but it stops one of the three axes from being a constant.

Anything that loads, names, or reports a deployed forecaster imports from here. There is
deliberately **no seed-agnostic accessor**: a call site that cannot say which seed it means
is a call site that was silently mixing three experiments.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = _REPO_ROOT / "checkpoints" / "lstm"

# The DQN training seeds, and the forecaster deployed with each. Trained by
# ``scripts.train_lstm --seed <s>`` on the A6.2 split (train SCN-01/02/03/10, val SCN-04,
# test SCN-05 sealed); freeze-gate verdicts in ``checkpoints/lstm/<version>__report.json``
# and the rule-level verdicts in ``checkpoints/lstm/retrain_rule_verdict__{legacy,current}.json``
# (A6.3: the pre-split runs are the pre-registered primary, these are the declared secondary).
OFFICIAL_LSTM_BY_SEED: dict[int, str] = {
    42: "lstm__data-222f7d7e77a3__lstm-4eea979501dd.pt",
    123: "lstm__data-222f7d7e77a3__lstm-436e1a9a9af0.pt",
    2024: "lstm__data-222f7d7e77a3__lstm-3c088813becc.pt",
}

OFFICIAL_TRAIN_SEEDS: tuple[int, ...] = tuple(sorted(OFFICIAL_LSTM_BY_SEED))


def _require_seed(seed: int) -> str:
    """The pinned filename for ``seed``, or a loud error naming the seeds that exist."""
    try:
        return OFFICIAL_LSTM_BY_SEED[seed]
    except KeyError:
        raise KeyError(
            f"no official forecaster pinned for training seed {seed!r}; pinned seeds are "
            f"{list(OFFICIAL_TRAIN_SEEDS)}. A6.4 pairs each DQN training seed to the forecaster "
            f"of the same seed - if you are adding a training seed, train its forecaster and pin "
            f"it here rather than borrowing another seed's."
        ) from None


def official_lstm_filename(seed: int) -> str:
    """The deployed forecaster's filename for one DQN training seed."""
    return _require_seed(seed)


def official_lstm_path(seed: int) -> Path:
    """The deployed forecaster's path for one training seed, **unchecked**.

    Prefer :func:`official_lstm_checked`. This exists for reporting call sites that need the
    name without loading the file; anything that LOADS the checkpoint must go through the
    checked accessor, because a guard placed beside a bare constant is a guard someone forgets.
    """
    return CKPT_DIR / _require_seed(seed)


def official_lstm_version(seed: int) -> str:
    """The ``lstm-<hash>`` id of the forecaster deployed with ``seed``.

    Derived rather than restated so the id can never drift from the file actually loaded.
    """
    return _parse_version(_require_seed(seed), "lstm-")


def official_lstm_versions() -> dict[int, str]:
    """``{seed: lstm-<hash>}`` for every pinned seed - for reports that span all three arms."""
    return {seed: official_lstm_version(seed) for seed in OFFICIAL_TRAIN_SEEDS}


def official_lstm_checked(seed: int) -> Path:
    """The deployed forecaster for ``seed``, but only after proving the pins match the corpus.

    Prefer this over the path helpers at every LOAD site. Making the checked accessor the only
    convenient way to load a forecaster puts the invariant where the action is.
    """
    assert_official_matches_corpus()
    return official_lstm_path(seed)


def assert_official_matches_corpus(data_dir: Path | None = None) -> None:
    """Raise unless every pinned forecaster was trained on the corpus currently on disk.

    The single-pin design fixed forecaster identity drifting across five files. It does NOT catch
    the other half of the problem: the corpus underneath the pin being regenerated while the pin
    stays put. That is a silent failure - every run keeps loading a real checkpoint that simply no
    longer corresponds to ``data/lstm/``, and every results row records a data_version that was
    true yesterday.

    Called by the campaign entry points so a stale pin stops the run instead of quietly poisoning
    a provenance chain. Cheap: it reads one manifest and parses three filenames.

    All three pins are checked in one call rather than only the seed being loaded. A guard that
    validated just the arm you happen to be running would pass a matrix whose other two cells are
    stale, which is exactly the partial-update failure this module exists to prevent.
    """
    from scripts.train_lstm import _dataset_data_version  # local: avoids a package import cycle

    unpinned = [s for s in OFFICIAL_TRAIN_SEEDS if OFFICIAL_LSTM_BY_SEED[s] == "PENDING"]
    if unpinned:
        raise RuntimeError(
            f"OFFICIAL forecaster is unpinned for seed(s) {unpinned}. Train them "
            f"(python -m scripts.train_lstm --seed <s>) and set OFFICIAL_LSTM_BY_SEED in "
            f"src/provenance/official.py. Refusing to run on a placeholder."
        )

    root = data_dir if data_dir is not None else _REPO_ROOT / "data" / "lstm"
    on_disk = _dataset_data_version(root)
    for seed in OFFICIAL_TRAIN_SEEDS:
        filename = OFFICIAL_LSTM_BY_SEED[seed]
        pinned = _parse_version(filename, "data-")
        if on_disk != pinned:
            raise RuntimeError(
                f"OFFICIAL forecaster is stale for training seed {seed}.\n"
                f"  pinned  (src/provenance/official.py): {pinned}\n"
                f"  on disk ({root}):                     {on_disk}\n"
                f"The pinned checkpoint {filename!r} was trained on a corpus that is no "
                f"longer the one on disk. Retrain the forecaster and re-pin "
                f"OFFICIAL_LSTM_BY_SEED, or restore the old corpus. Refusing to run rather than "
                f"record a data_version that is not true."
            )
        if not (CKPT_DIR / filename).exists():
            raise FileNotFoundError(
                f"OFFICIAL forecaster for training seed {seed} is pinned to {filename!r}, which is "
                f"not in {CKPT_DIR} (checkpoints are gitignored - regenerate with "
                f"python -m scripts.train_lstm --seed {seed})."
            )


def assert_official_matches_split(data_dir: Path | None = None) -> None:
    """Raise unless every pinned checkpoint was trained on the CURRENT ``SPLITS``.

    The corpus hash covers all 100 CSVs whichever subset is actually read, so changing the split
    leaves ``data_version`` untouched: a pin can go stale in a way ``assert_official_matches_corpus``
    is structurally unable to see. A6.4 put the split inside the hashed training config for exactly
    this reason, and this reads it back out of the checkpoints so the invariant is enforced, not
    merely recorded. Separate from the corpus guard because it must open three checkpoint files.
    """
    import torch

    from src.ml.lstm_data import SPLITS

    assert_official_matches_corpus(data_dir)
    live = {k: list(v) for k, v in sorted(SPLITS.items())}
    for seed in OFFICIAL_TRAIN_SEEDS:
        path = official_lstm_path(seed)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        recorded = ckpt.get("config", {}).get("splits")
        if recorded is None:
            raise RuntimeError(
                f"{path.name} predates A6.4 and records no split, so it cannot be proven to match "
                f"the current SPLITS. Retrain seed {seed} rather than assume."
            )
        if recorded != live:
            raise RuntimeError(
                f"OFFICIAL forecaster for training seed {seed} was trained on a DIFFERENT split.\n"
                f"  checkpoint {path.name}: {recorded}\n"
                f"  src/ml/lstm_data.py:    {live}\n"
                f"The corpus hash cannot see this - it covers every CSV whichever subset is read. "
                f"Retrain and re-pin, or restore the split."
            )


def official_data_version() -> str:
    """The ``data-<hash>`` id of the corpus the deployed forecasters were trained on.

    All pinned seeds must agree: they are three draws of one training process over one corpus,
    and a disagreement means the pins were updated piecemeal.
    """
    versions = {seed: _parse_version(_require_seed(seed), "data-") for seed in OFFICIAL_TRAIN_SEEDS}
    distinct = set(versions.values())
    if len(distinct) != 1:
        raise RuntimeError(
            f"the pinned forecasters disagree about their corpus: {versions}. They must be three "
            f"seeds of ONE training process over ONE corpus; a split here means the pins were "
            f"updated one at a time."
        )
    return distinct.pop()


def _parse_version(filename: str, prefix: str) -> str:
    """Pull the ``<prefix><hash>`` component out of a checkpoint filename."""
    stem = Path(filename).stem
    for part in stem.split("__"):
        if part.startswith(prefix):
            return part
    raise ValueError(f"cannot parse a {prefix!r} version out of {filename!r}")
