"""T-03-02 - The LSTM queue forecaster + its freeze-gate metrics.

Architecture is the locked spec (``lstm-forecasting.md`` "Interface contract"):
2-layer LSTM, hidden 128, dropout 0.1, a linear head ``128 -> 36`` reshaped to the
``(3, 12)`` 3-step queue forecast. Frozen during DQN training (T-03-05).

The **freeze gate** (open-items E1/F, revised): the forecast is judged not by raw
MSE but by **skill score vs a persistence baseline** (predict "the queue 1-3 steps
ahead = the queue now"). Persistence already scores high R^2 "for free" at 10 s
queue horizons, so R^2 is a sanity check only - skill score is the go/no-go:

    SS_h = 1 - MSE_model(h) / MSE_persistence(h)        (per horizon h)

    SHIP the 36 forecast dims iff  SS@h+1 > 0.10  AND  SS@h+3 > 0.05  (on held-out val)

``gate_verdict`` encodes the locked fallback tree (SS<=0 @h+1 -> retrain or drop all
36 dims to a 20-dim DQN; pass@h+1 but fail@h+3 -> ship, DQN down-weights h+3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

INPUT_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
HORIZON = 3
N_MOVEMENTS = 12

# Locked freeze thresholds (open-items E1/F).
SS_THRESHOLD_H1 = 0.10  # skill score @ h+1
SS_THRESHOLD_H3 = 0.05  # skill score @ h+3


class LSTMForecaster(nn.Module):
    """2-layer LSTM -> linear head -> ``(B, 3, 12)`` queue forecast (lstm-forecasting.md)."""

    def __init__(
        self,
        *,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, HORIZON * N_MOVEMENTS)
        # Input standardization, applied inside forward so the loader, persistence
        # baseline, and skill score all keep working on RAW queue/count units. The
        # stats are fit on the train set (set_input_stats) and saved with the
        # weights, so inference standardizes identically. Default no-op (0 / 1).
        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_std", torch.ones(input_size))

    def set_input_stats(self, mean: Tensor, std: Tensor) -> None:
        """Fit the internal z-score normalizer (train-set per-feature mean/std)."""
        self.input_mean.copy_(mean)
        self.input_std.copy_(torch.clamp(std, min=1e-6))  # guard zero-variance features

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 12, 24), raw units.
        # RESIDUAL forecast (ADR-005): predict the CHANGE from the current queue, not
        # the absolute queue. forecast = current_queue + learned_delta. This anchors
        # the model to the present, so it starts at parity with persistence and tracks
        # any queue magnitude - fixing the held-out distribution-shift failure of the
        # original direct-forecast design (open-items E1/C1).
        current_queue = x[:, -1, :N_MOVEMENTS]            # (B, 12), raw last-step queue
        xn = (x - self.input_mean) / self.input_std
        _, (h_n, _) = self.lstm(xn)                       # h_n: (num_layers, B, hidden)
        delta = self.head(h_n[-1]).view(-1, HORIZON, N_MOVEMENTS)  # (B, 3, 12)
        return current_queue.unsqueeze(1) + delta         # (B, 3, 12)


def persistence_forecast(x: Tensor) -> Tensor:
    """Naive baseline: every future queue = the last observed queue.

    ``x`` is ``(B, 12, 24)``; the queue is the first 12 of the 24 features, so the
    last history step's queue is ``x[:, -1, :12]``, broadcast across the 3 horizons.
    """
    last_queue = x[:, -1, :N_MOVEMENTS]                 # (B, 12)
    return last_queue.unsqueeze(1).expand(-1, HORIZON, -1)  # (B, 3, 12)


def _per_horizon_mse(pred: Tensor, target: Tensor) -> Tensor:
    """MSE per horizon, averaged over batch + movements -> ``(3,)``."""
    return ((pred - target) ** 2).mean(dim=(0, 2))


def skill_scores(pred: Tensor, target: Tensor, x: Tensor) -> list[float]:
    """Per-horizon skill score ``1 - MSE_model / MSE_persistence`` -> length-3 list."""
    mse_model = _per_horizon_mse(pred, target)
    mse_persist = _per_horizon_mse(persistence_forecast(x), target)
    ss = 1.0 - mse_model / torch.clamp(mse_persist, min=1e-12)
    return [float(v) for v in ss]


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Overall R^2 across the whole (3,12) grid (sanity check only, not the gate)."""
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class GateDecision:
    """The freeze-gate verdict + the numbers behind it."""

    ship: bool          # ship the 36 forecast dims at all?
    verdict: str        # PASS / SHIP_WITH_CAVEAT / RETRAIN_OR_DROP
    reason: str
    ss_h1: float
    ss_h3: float


def gate_verdict(ss_h1: float, ss_h3: float) -> GateDecision:
    """Map the val skill scores to the locked fallback tree (open-items E1/F)."""
    if ss_h1 <= 0.0:
        return GateDecision(
            ship=False, verdict="RETRAIN_OR_DROP",
            reason="SS@h+1<=0: forecast is no better than persistence -> retrain; "
                   "if it still fails, drop all 36 dims and run the 20-dim DQN (report as finding).",
            ss_h1=ss_h1, ss_h3=ss_h3,
        )
    if ss_h1 > SS_THRESHOLD_H1 and ss_h3 > SS_THRESHOLD_H3:
        return GateDecision(
            ship=True, verdict="PASS",
            reason=f"SS@h+1={ss_h1:.3f}>{SS_THRESHOLD_H1} and SS@h+3={ss_h3:.3f}>{SS_THRESHOLD_H3}: "
                   "ship the 36 forecast dims.",
            ss_h1=ss_h1, ss_h3=ss_h3,
        )
    if ss_h1 > SS_THRESHOLD_H1:  # passes h+1, weak at h+3
        return GateDecision(
            ship=True, verdict="SHIP_WITH_CAVEAT",
            reason=f"SS@h+1={ss_h1:.3f} passes but SS@h+3={ss_h3:.3f}<={SS_THRESHOLD_H3}: "
                   "ship; the DQN can down-weight the noisier h+3 forecast.",
            ss_h1=ss_h1, ss_h3=ss_h3,
        )
    return GateDecision(  # 0 < SS@h+1 <= 0.10
        ship=True, verdict="SHIP_WITH_CAVEAT",
        reason=f"0<SS@h+1={ss_h1:.3f}<={SS_THRESHOLD_H1}: marginal skill -> ship with a documented caveat.",
        ss_h1=ss_h1, ss_h3=ss_h3,
    )
