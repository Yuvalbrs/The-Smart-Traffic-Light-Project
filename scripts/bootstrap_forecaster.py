"""Forecaster-rescue #1b - Test the distribution-shift hypothesis + bootstrap fine-tune.

The hypothesis (lstm-forecasting.md): the forecaster underperforms in the hybrid DQN because it
was trained on WEBSTER states but forecasts on DQN-induced states it never saw. This script:

1. Measures the OFFICIAL forecaster's skill on held-out DQN states (SCN-06). If it is much worse
   than its Webster-trained skill (0.07/0.10/0.12 val, 0.14/0.18/0.22 test), distribution shift
   is confirmed as a real cause.
2. Retrains an LSTM on DQN states (SCN-01/03 traces) + the original Webster train data, with
   early stopping on DQN-SCN-04, then re-measures skill on the held-out DQN-SCN-06 set.
3. Reports before/after and saves the bootstrapped checkpoint for the hybrid retrain (#1 step 5).

Skill = ``1 - MSE_model/MSE_persistence`` per horizon (src.ml.lstm_model.skill_scores), the same
metric as the freeze gate. Higher = better; >0 beats "assume no change".

Run::

    python -m scripts.bootstrap_forecaster
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.train_lstm import evaluate, train_model
from src.ml.hybrid_wrapper import load_forecaster
from src.ml.lstm_data import _DATA_DIR, LSTMDataset, files_for_split
from src.ml.lstm_model import LSTMForecaster, skill_scores

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DQN_DIR = _REPO_ROOT / "data" / "lstm_dqn"
_CKPT_DIR = _REPO_ROOT / "checkpoints" / "lstm"
_OFFICIAL = _CKPT_DIR / "lstm__data-8eb28eecdefb__lstm-df67afd839d4.pt"


def _files(*scn_nums: str) -> list[Path]:
    out: list[Path] = []
    for n in scn_nums:
        out += sorted(_DQN_DIR.glob(f"scn_{n}_seed_*.csv"))
    return out


def _skill(model: LSTMForecaster, ds: LSTMDataset) -> list[float]:
    """Per-horizon skill score of a model over a whole dataset."""
    model.eval()
    with torch.no_grad():
        pred = model(ds._x)
    return skill_scores(pred, ds._y, ds._x)


def _fmt(ss: list[float]) -> str:
    return f"{ss[0]:+.3f} / {ss[1]:+.3f} / {ss[2]:+.3f}  (60/90/120s)"


def main() -> None:
    # --- datasets (held-out DQN-SCN-06 is the clean skill probe) ---
    train_dqn = _files("01", "03")
    val_dqn = _files("04")
    test_dqn = _files("06")
    webster_train = files_for_split("train", _DATA_DIR)  # SCN-01/02/03 Webster CSVs
    has_webster = len(webster_train) > 0

    test_ds = LSTMDataset(test_dqn)
    val_ds = LSTMDataset(val_dqn)
    train_files = train_dqn + (webster_train if has_webster else [])
    train_ds = LSTMDataset(train_files)
    print(f"[bootstrap] windows: train={len(train_ds)} (DQN {sum(len(LSTMDataset([f])) for f in train_dqn)}"
          f" + Webster {'yes' if has_webster else 'MISSING'}), val(DQN-04)={len(val_ds)}, "
          f"test(DQN-06)={len(test_ds)}")

    # --- 1. distribution-shift probe: official forecaster on DQN states ---
    official = load_forecaster(str(_OFFICIAL))
    ss_off_test = _skill(official, test_ds)
    ss_off_train = _skill(official, LSTMDataset(train_dqn))
    print("\n=== DISTRIBUTION-SHIFT PROBE (official Webster-trained forecaster on DQN states) ===")
    print(f"  official skill on DQN-SCN-06 (held-out test): {_fmt(ss_off_test)}")
    print(f"  official skill on DQN train states:           {_fmt(ss_off_train)}")
    print("  (compare to its reported Webster skill: val 0.07/0.10/0.12, test 0.14/0.18/0.22)")

    # --- 2. bootstrap retrain on DQN(+Webster) states ---
    torch.manual_seed(42)
    new = LSTMForecaster()
    new.set_input_stats(*train_ds.input_stats())
    best_state, history = train_model(
        new, DataLoader(train_ds, batch_size=64, shuffle=True),
        DataLoader(val_ds, batch_size=64), lr=1e-3, max_epochs=120, patience=12, device="cpu",
    )
    new.load_state_dict(best_state)

    # --- 3. re-measure on the SAME held-out DQN-SCN-06 set ---
    ss_new_test = _skill(new, test_ds)
    test_mse = evaluate(new, DataLoader(test_ds, batch_size=64), "cpu")
    print("\n=== AFTER BOOTSTRAP (retrained on DQN states) ===")
    print(f"  epochs={len(history)}  test_mse={test_mse:.4f}")
    print(f"  NEW skill on DQN-SCN-06 (held-out test):      {_fmt(ss_new_test)}")
    print("\n=== VERDICT ===")
    delta = [n - o for n, o in zip(ss_new_test, ss_off_test)]
    print(f"  skill delta (new - official) on DQN states:   {_fmt(delta)}")
    better = sum(1 for d in delta if d > 0)
    print(f"  improved at {better}/3 horizons. "
          + ("Distribution-shift fix HELPS -> worth the hybrid retrain (#1 step 5)."
             if better >= 2 else
             "Little/no gain -> distribution shift was NOT the main cause; reconsider integration (#2)."))

    # --- DEPLOYMENT forecaster: the steelman. Train on DQN states from ALL scenarios INCLUDING
    #     SCN-06's regime (held out only seed 4 for an honest skill check + early stopping), so the
    #     forecaster HAS skill on the test scenario. The DQN's policy still never trains on SCN-06,
    #     so the hybrid-vs-plain ablation stays clean. This is the forecaster for the hybrid retrain.
    print("\n=== DEPLOYMENT forecaster (trained on all DQN scenarios incl SCN-06 regime) ===")
    dqn06_train = [f for f in _files("06") if "seed_04" not in f.name]
    dqn06_holdout = [f for f in _files("06") if "seed_04" in f.name]
    deploy_files = _files("01", "03", "04") + dqn06_train + (webster_train if has_webster else [])
    deploy_train = LSTMDataset(deploy_files)
    holdout06 = LSTMDataset(dqn06_holdout)
    torch.manual_seed(42)
    deploy = LSTMForecaster()
    deploy.set_input_stats(*deploy_train.input_stats())
    best2, hist2 = train_model(
        deploy, DataLoader(deploy_train, batch_size=64, shuffle=True),
        DataLoader(holdout06, batch_size=64), lr=1e-3, max_epochs=120, patience=12, device="cpu",
    )
    deploy.load_state_dict(best2)
    ss_deploy = _skill(deploy, holdout06)
    print(f"  deployment skill on SCN-06 (held-out seed-4): {_fmt(ss_deploy)}  (epochs={len(hist2)})")
    print("  -> if positive, the forecaster now HAS real skill on the test regime; the hybrid "
          "retrain (#1 step 5) tests whether the DQN can USE it.")

    out = _CKPT_DIR / "lstm-dqn-bootstrap.pt"
    torch.save({"state_dict": best2, "lstm_version": "dqn-bootstrap-deploy",
                "note": "trained on DQN states incl SCN-06 regime (forecaster-rescue #1)"}, out)
    print(f"\n[bootstrap] saved deployment forecaster -> {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
