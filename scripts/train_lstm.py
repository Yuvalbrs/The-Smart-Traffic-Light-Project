"""T-03-02 - Train the frozen LSTM queue forecaster + run the freeze gate.

Trains :class:`LSTMForecaster` on the windowed CSVs (T-03-01 loader): MSE loss,
Adam lr=1e-3, batch 64, early stopping patience 10 on val MSE
(``lstm-forecasting.md`` "Training loop"). Saves the best-by-val checkpoint with a
provenance filename embedding the version chain, then evaluates the **freeze gate**
(skill score vs persistence on held-out val) and the standalone test MSE.

The checkpoint + a JSON report land in ``checkpoints/lstm/`` (gitignored). The
report records the gate verdict so the go/no-go is auditable, not a vibe.

Run::

    python -m scripts.train_lstm --seed 42
    python -m scripts.train_lstm --seed 42 --max-epochs 5     # quick
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.lstm_data import _DATA_DIR, make_dataloaders
from src.ml.lstm_model import (
    DROPOUT,
    HIDDEN_SIZE,
    HORIZON,
    INPUT_SIZE,
    NUM_LAYERS,
    GateDecision,
    LSTMForecaster,
    gate_verdict,
    r2_score,
    skill_scores,
)
from src.provenance.versions import (
    checkpoint_filename,
    config_hash,
    git_sha,
    lstm_version,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CKPT_DIR = _REPO_ROOT / "checkpoints" / "lstm"


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Mean MSE over a loader (the early-stopping signal + standalone metric)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += float(((pred - y) ** 2).sum())
            n += y.numel()
    return total / n if n else float("nan")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[dict, list[dict]]:
    """Train with early stopping on val MSE. Returns ``(best_state, history)``."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

        val_mse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "val_mse": val_mse})
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:  # early stopping
                break

    return best_state, history


def run_freeze_gate(model: nn.Module, val_loader: DataLoader, device: str) -> tuple[GateDecision, dict]:
    """Compute per-horizon skill score + R^2 on val and apply the gate."""
    model.eval()
    xs, preds, ys = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            xs.append(x.cpu())
            preds.append(model(x).cpu())
            ys.append(y)
    x_all, pred_all, y_all = torch.cat(xs), torch.cat(preds), torch.cat(ys)
    ss = skill_scores(pred_all, y_all, x_all)  # [SS@h+1, SS@h+2, SS@h+3]
    decision = gate_verdict(ss[0], ss[HORIZON - 1])
    metrics = {"skill_scores": ss, "r2": r2_score(pred_all, y_all)}
    return decision, metrics


def _dataset_data_version(data_dir: Path) -> str:
    """Aggregate the 50 per-file data_versions (manifest) into one dataset version."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        dvs = sorted(e["data_version"] for e in json.loads(manifest_path.read_text()))
        return f"data-{config_hash({'file_data_versions': dvs})}"
    return f"data-{config_hash({'data_dir': str(data_dir)})}"  # fallback, no manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # deterministic init + shuffling
    loaders = make_dataloaders(_DATA_DIR, batch_size=args.batch)
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"[lstm-train] windows: {sizes}")

    model = LSTMForecaster()
    # Fit the input normalizer on the TRAIN windows only (no val/test leakage),
    # then train; the stats ride in the checkpoint for identical inference scaling.
    train_x = loaders["train"].dataset._x  # (N, 12, 24) raw features
    model.set_input_stats(train_x.mean(dim=(0, 1)), train_x.std(dim=(0, 1)))
    best_state, history = train_model(
        model, loaders["train"], loaders["val"],
        lr=args.lr, max_epochs=args.max_epochs, patience=args.patience, device=args.device,
    )
    model.load_state_dict(best_state)

    val_mse = evaluate(model, loaders["val"], args.device)
    test_mse = evaluate(model, loaders["test"], args.device)
    decision, gate_metrics = run_freeze_gate(model, loaders["val"], args.device)

    # provenance chain
    lstm_config = {
        "input_size": INPUT_SIZE, "hidden": HIDDEN_SIZE, "layers": NUM_LAYERS,
        "dropout": DROPOUT, "horizon": HORIZON, "lr": args.lr, "batch": args.batch,
        "max_epochs": args.max_epochs, "patience": args.patience,
    }
    data_v = _dataset_data_version(_DATA_DIR)
    lstm_v = lstm_version(
        data_version=data_v,
        lstm_config_hash=config_hash(lstm_config),
        training_code_git_sha=git_sha() or "unknown",
        training_seed=args.seed,
    )

    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = _CKPT_DIR / checkpoint_filename("lstm", data_version=data_v, lstm_version=lstm_v)
    torch.save(
        {"state_dict": best_state, "lstm_version": lstm_v, "data_version": data_v,
         "config": lstm_config, "seed": args.seed},
        ckpt_path,
    )
    report = {
        "lstm_version": lstm_v, "data_version": data_v, "seed": args.seed,
        "window_sizes": sizes, "epochs_run": len(history),
        "val_mse": val_mse, "test_mse": test_mse,
        "skill_scores_val": gate_metrics["skill_scores"], "r2_val": gate_metrics["r2"],
        "gate": asdict(decision), "checkpoint": ckpt_path.name,
    }
    (_CKPT_DIR / f"{lstm_v}__report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    ss = gate_metrics["skill_scores"]
    print(f"[lstm-train] epochs={len(history)} val_mse={val_mse:.4f} test_mse={test_mse:.4f} r2={gate_metrics['r2']:.3f}")
    print(f"[lstm-train] skill: SS@h+1={ss[0]:.3f} SS@h+2={ss[1]:.3f} SS@h+3={ss[2]:.3f}")
    print(f"[lstm-train] FREEZE GATE: {decision.verdict} (ship={decision.ship})")
    print(f"             {decision.reason}")
    print(f"[lstm-train] OK -> {ckpt_path.relative_to(_REPO_ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
