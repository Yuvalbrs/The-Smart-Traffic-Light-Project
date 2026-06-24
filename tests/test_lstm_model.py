"""T-03-02 - Tests for the LSTM forecaster, its metrics, and the freeze gate.

Model shape, the persistence/skill-score math (hand-verifiable), the gate's
fallback-tree mapping, and a short end-to-end training smoke on tiny in-memory
loaders (no SUMO, no real CSVs) that asserts the loss actually goes down.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_lstm import evaluate, run_freeze_gate, train_model
from src.ml.lstm_model import (
    HORIZON,
    N_MOVEMENTS,
    LSTMForecaster,
    gate_verdict,
    persistence_forecast,
    skill_scores,
)


def test_forward_shape() -> None:
    model = LSTMForecaster()
    out = model(torch.randn(4, 12, 24))
    assert tuple(out.shape) == (4, HORIZON, N_MOVEMENTS)  # (4, 3, 12)


def test_persistence_uses_last_observed_queue() -> None:
    x = torch.zeros(2, 12, 24)
    x[:, -1, :N_MOVEMENTS] = 7.0  # last step's queue = 7 for every movement
    p = persistence_forecast(x)
    assert tuple(p.shape) == (2, HORIZON, N_MOVEMENTS)
    assert torch.all(p == 7.0)  # every horizon predicts the last observed queue


def test_skill_score_endpoints() -> None:
    """Perfect prediction -> SS=1; predicting persistence -> SS=0."""
    x = torch.randn(8, 12, 24)
    target = torch.randn(8, HORIZON, N_MOVEMENTS)
    assert all(abs(s - 1.0) < 1e-5 for s in skill_scores(target, target, x))  # perfect
    persist = persistence_forecast(x)
    assert all(abs(s) < 1e-5 for s in skill_scores(persist, target, x))       # = baseline


def test_gate_fallback_tree() -> None:
    assert gate_verdict(0.20, 0.10).verdict == "PASS"
    assert gate_verdict(0.20, 0.01).verdict == "SHIP_WITH_CAVEAT"   # weak h+3
    assert gate_verdict(0.05, 0.02).verdict == "SHIP_WITH_CAVEAT"   # marginal h+1
    bad = gate_verdict(-0.1, 0.2)
    assert bad.verdict == "RETRAIN_OR_DROP" and bad.ship is False


def _toy_loaders():
    """A tiny learnable task: target = first 3x12 input features of the last step."""
    torch.manual_seed(0)
    x = torch.randn(128, 12, 24)
    y = x[:, -1, :N_MOVEMENTS].unsqueeze(1).repeat(1, HORIZON, 1) + 0.01 * torch.randn(128, HORIZON, N_MOVEMENTS)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


def test_training_reduces_loss_and_saves_state() -> None:
    train_loader, val_loader = _toy_loaders()
    model = LSTMForecaster()
    before = evaluate(model, val_loader, "cpu")
    best_state, history = train_model(
        model, train_loader, val_loader, lr=1e-2, max_epochs=15, patience=5
    )
    model.load_state_dict(best_state)
    after = evaluate(model, val_loader, "cpu")
    assert after < before          # it actually learned
    assert len(history) >= 1
    decision, metrics = run_freeze_gate(model, val_loader, "cpu")
    assert len(metrics["skill_scores"]) == HORIZON
    assert decision.verdict in {"PASS", "SHIP_WITH_CAVEAT", "RETRAIN_OR_DROP"}
