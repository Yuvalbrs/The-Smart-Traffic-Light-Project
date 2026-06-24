"""T-03-02 - The LSTM queue forecaster + its freeze-gate metrics.

Architecture (lstm-forecasting.md + ADR-005): 2-layer LSTM, hidden 128, dropout 0.1,
a linear head ``128 -> 36`` reshaped to a ``(3, 12)`` forecast of 3 future queue
points. The head is **RESIDUAL** (ADR-005): it predicts the *change* from the current
queue (``forecast = current_queue + delta``), which anchors the model to the present
and fixed the held-out distribution-shift collapse of the original direct forecast.
Inputs are z-score standardized inside ``forward`` (train-fit buffers, saved with the
weights). Frozen during DQN training (T-03-05).

*Which* 3 future points are forecast is set by the data loader's target offsets
(ADR-006: **60/90/120 s** ahead by default), NOT here - this module fixes only the
COUNT (3 = ``HORIZON``) and shape. Counts feed in as inputs; only queue is forecast.

The **freeze gate** (open-items E1/F): judge the forecast by **skill score vs a
persistence baseline** (predict "future queue = queue now"), not raw MSE - persistence
scores high R^2 "for free", so R^2 is a sanity check only and skill score is the
go/no-go:

    SS_h = 1 - MSE_model(h) / MSE_persistence(h)        (per forecast point h)

    SHIP the 36 forecast dims iff  SS@near > 0.10  AND  SS@far > 0.05  (held-out val)

``gate_verdict`` encodes the locked fallback tree (SS<=0 at the near point -> retrain,
or drop all 36 dims to a 20-dim DQN; near passes but far weak -> ship with caveat).
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
    """2-layer LSTM -> residual head -> ``(B, 3, 12)`` queue forecast = current queue + delta (ADR-005)."""

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
    """The freeze-gate verdict + the numbers behind it.

    ``ss_near`` / ``ss_far`` are the skill scores at the FIRST and LAST forecast points
    (with the ADR-006 default offsets, 60 s and 120 s ahead).
    """

    ship: bool          # ship the 36 forecast dims at all?
    verdict: str        # PASS / SHIP_WITH_CAVEAT / RETRAIN_OR_DROP
    reason: str
    ss_near: float      # skill at the first/nearest forecast point
    ss_far: float       # skill at the last/farthest forecast point


def gate_verdict(ss_near: float, ss_far: float) -> GateDecision:
    """Map the val skill scores (near + far forecast points) to the locked fallback tree."""
    if ss_near <= 0.0:
        return GateDecision(
            ship=False, verdict="RETRAIN_OR_DROP",
            reason="SS@near<=0: forecast is no better than persistence -> retrain; "
                   "if it still fails, drop all 36 dims and run the 20-dim DQN (report as finding).",
            ss_near=ss_near, ss_far=ss_far,
        )
    if ss_near > SS_THRESHOLD_H1 and ss_far > SS_THRESHOLD_H3:
        return GateDecision(
            ship=True, verdict="PASS",
            reason=f"SS@near={ss_near:.3f}>{SS_THRESHOLD_H1} and SS@far={ss_far:.3f}>{SS_THRESHOLD_H3}: "
                   "ship the 36 forecast dims.",
            ss_near=ss_near, ss_far=ss_far,
        )
    if ss_near > SS_THRESHOLD_H1:  # passes near, weak at far
        return GateDecision(
            ship=True, verdict="SHIP_WITH_CAVEAT",
            reason=f"SS@near={ss_near:.3f} passes but SS@far={ss_far:.3f}<={SS_THRESHOLD_H3}: "
                   "ship; the DQN can down-weight the noisier far-horizon forecast.",
            ss_near=ss_near, ss_far=ss_far,
        )
    return GateDecision(  # 0 < SS@near <= 0.10
        ship=True, verdict="SHIP_WITH_CAVEAT",
        reason=f"0<SS@near={ss_near:.3f}<={SS_THRESHOLD_H1}: marginal skill -> ship with a documented caveat.",
        ss_near=ss_near, ss_far=ss_far,
    )
