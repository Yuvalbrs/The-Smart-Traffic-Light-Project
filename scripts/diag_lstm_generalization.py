"""Diagnostic: is the forecaster's training broken, or is forecasting just hard here?

The decisive observation: the UNTRAINED model already scores val_mse ~2.18, and no epoch of
training ever gets below that. With a residual head (ADR-005: forecast = current_queue + delta)
an untrained model is approximately PERSISTENCE, so "training never beats the untrained model"
means gradient descent is moving the delta head AWAY from zero and being punished for it.

This does NOT tune anything for deployment. It establishes whether lr=1e-3 diverges, by
tracking train AND val together across a few learning rates. A diverging optimiser shows val
rising with train falling (overfit) or BOTH bouncing (step too large). Whatever it shows, the
forecaster actually deployed is chosen by a rule committed before its skill is measured.
"""
from __future__ import annotations

import torch

from scripts.train_lstm import evaluate
from src.ml.lstm_data import DEFAULT_TARGET_OFFSETS, make_dataloaders
from src.ml.lstm_model import LSTMForecaster

EPOCHS = 8
LRS = [1e-3, 3e-4, 1e-4, 3e-5]


def run(lr: float, dl) -> None:
    torch.manual_seed(42)
    model = LSTMForecaster()
    base_tr = evaluate(model, dl["train"], "cpu")
    base_va = evaluate(model, dl["val"], "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    print(f"\n--- lr={lr:g} ---")
    print(f"  epoch    train_mse      val_mse")
    print(f"  {'init':>5}   {base_tr:10.5f}   {base_va:10.5f}   <- UNTRAINED (residual head ~ persistence)")
    best = base_va
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl["train"]:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        tr = evaluate(model, dl["train"], "cpu")
        va = evaluate(model, dl["val"], "cpu")
        best = min(best, va)
        flag = "  <- beats untrained" if va < base_va else ""
        print(f"  {ep:>5}   {tr:10.5f}   {va:10.5f}{flag}")
    verdict = "IMPROVES on untrained" if best < base_va else "NEVER beats untrained"
    print(f"  => best val {best:.5f} vs untrained {base_va:.5f}  :: {verdict}")


def main() -> None:
    dl = make_dataloaders(batch_size=64, target_offsets=DEFAULT_TARGET_OFFSETS)
    for lr in LRS:
        run(lr, dl)


if __name__ == "__main__":
    main()
