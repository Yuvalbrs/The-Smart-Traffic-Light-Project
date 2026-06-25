"""T-03-05 - HybridStateWrapper: concatenate the frozen LSTM forecast onto the DQN state.

The locked hybrid design (Option A, hybrid-integration.md / ADR-004): at every decision step
the frozen forecaster turns the last 12 steps of per-movement (queue, count) features into a
``(3, 12)`` queue forecast; we flatten it to 36 dims and append it to the 20-dim base state ->
a 56-dim observation. The DQN class is unchanged (its ``obs_dim`` just becomes 56); the wrapper
is a composable layer so the forecast-blind ablation simply trains on the bare ``SUMOEnv``.

Build-time corrections folded in (hybrid-integration.md AMENDMENT 2026-06-24, ADR-005/006):

* **Forecast normalization is z-score, not ``/30``+clip.** The old ``/30`` assumed a 30 s horizon
  with queues in ``[0, 30]``; at the ADR-006 60/90/120 s horizon the residual head (ADR-005)
  emits *absolute* queue forecasts that exceed 100 in heavy scenarios, so ``/30``+``clip(0,1)``
  pins everything at 1.0 and destroys the signal. Instead we standardize each movement's
  forecast by the train-set **queue** mean/std the forecaster already carries in its
  ``input_mean/input_std`` buffers (the first 12 of the 24 input features are queue). This is
  per-feature, never saturates, and rides in the checkpoint so inference scaling is identical.
  For a model with default (0/1) stats - e.g. the random-LSTM control - it is a no-op.
* **Real ``SUMOEnv`` API:** features come from ``env.movement_features()`` (returns
  ``(queue[12], count[12])``), and ``reset``/``step`` use the gymnasium 5-tuple / ``(obs, info)``
  contract. ``get_action_mask`` and ``movement_features`` forward to the base env unchanged via
  the gymnasium ``Wrapper`` attribute delegation.

Cold start (hybrid-integration.md "Cold start handling"): the first 11 augmentations of an
episode (history not yet 12 deep) get an all-zero forecast - an honest "no forecast available"
signal - rather than bootstrapping off a repeated first frame.
"""

from __future__ import annotations

import collections
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.ml.dqn import OBS_DIM
from src.ml.lstm_data import INPUT_LEN
from src.ml.lstm_model import HORIZON, N_MOVEMENTS, LSTMForecaster

FORECAST_DIM = HORIZON * N_MOVEMENTS  # 3 horizons x 12 movements = 36 flattened forecast dims
HYBRID_OBS_DIM = OBS_DIM + FORECAST_DIM  # 20 + 36 = 56 (locked)


def load_forecaster(
    checkpoint_path: str | Path, *, device: str = "cpu", freeze: bool = True
) -> LSTMForecaster:
    """Load a trained LSTM checkpoint into a frozen, eval-mode forecaster.

    The checkpoint (from ``scripts/train_lstm.py``) stores ``{"state_dict": ...}`` whose buffers
    include the fitted ``input_mean/input_std`` (set before training), so inference standardizes
    identically. ``freeze=True`` (the default, and the locked design: the forecaster is frozen
    during DQN training) sets ``eval()`` and ``requires_grad_(False)`` on every parameter.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LSTMForecaster()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


class HybridStateWrapper(gym.Wrapper):
    """Augment a ``SUMOEnv`` observation with the flattened, z-scored LSTM queue forecast.

    Parameters
    ----------
    env : gym.Env
        The base ``SUMOEnv`` (20-dim observations, ``movement_features()`` available).
    forecaster : LSTMForecaster
        The frozen forecaster (typically from :func:`load_forecaster`). Set to ``eval`` here.
    history_len : int, optional
        Steps of history fed to the LSTM. Default 12 (``INPUT_LEN``); must match how the
        forecaster was trained.
    """

    def __init__(
        self,
        env: gym.Env,
        forecaster: LSTMForecaster,
        *,
        history_len: int = INPUT_LEN,
    ) -> None:
        super().__init__(env)
        self.forecaster = forecaster.eval()
        self._history_len = history_len
        self._history: collections.deque[np.ndarray] = collections.deque(maxlen=history_len)

        base = env.observation_space
        fc_low = np.full(FORECAST_DIM, -np.inf, np.float32)
        fc_high = np.full(FORECAST_DIM, np.inf, np.float32)
        low = np.concatenate([np.asarray(base.low, dtype=np.float32), fc_low])
        high = np.concatenate([np.asarray(base.high, dtype=np.float32), fc_high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Per-movement queue mean/std for the forecast z-score: the first 12 of the 24 input
        # features are queue (lstm_data layout [q0..q11, c0..c11]), and the forecast is queue.
        qm = forecaster.input_mean[:N_MOVEMENTS].detach().cpu().numpy()
        qs = forecaster.input_std[:N_MOVEMENTS].detach().cpu().numpy()
        self._queue_mean = qm.astype(np.float32)
        self._queue_std = qs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the base env, clear history, and return the augmented ``(obs, info)``."""
        self._history.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action: int):
        """Step the base env and augment the next observation (gymnasium 5-tuple)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Forward the base env's legal-action mask unchanged (the wrapper never alters it)."""
        return self.env.get_action_mask()

    def movement_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Forward the base env's ``(queue, count)`` per-movement features."""
        return self.env.movement_features()

    def _augment(self, base_obs: np.ndarray) -> np.ndarray:
        """Append current features to history, run the forecaster, concat the 36 forecast dims."""
        queue, count = self.env.movement_features()  # (12,), (12,)
        self._history.append(np.concatenate([queue, count]).astype(np.float32))  # (24,) [q..,c..]
        if len(self._history) < self._history_len:  # cold start -> honest zero forecast
            forecast = np.zeros(FORECAST_DIM, dtype=np.float32)
        else:
            x = torch.from_numpy(np.stack(self._history)).unsqueeze(0)  # (1, 12, 24)
            with torch.no_grad():
                pred = self.forecaster(x).squeeze(0).cpu().numpy()  # (3, 12) raw queue forecast
            norm = (pred - self._queue_mean) / self._queue_std  # per-movement z-score, no clip
            forecast = norm.astype(np.float32).flatten()  # (36,)
        return np.concatenate([base_obs.astype(np.float32), forecast])
