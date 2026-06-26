"""Shared SUMOEnv construction for the training scripts (T-03-06/07/08).

One place that turns ``(scenario_id, route_seed)`` into a ready-to-drive env, so
``train_dqn`` and ``sanity_gate`` cannot drift on the locked params (decision interval,
switch penalty) or on how the hybrid forecast wrapper is attached. In particular
``switch_penalty`` is threaded through from the run config - never hardcoded - so the
T-04-03 lambda sweep ({0, 0.05, 0.1, 0.5}) actually changes the env reward and matches
the value recorded in ``config.yaml``.

Lives in the scripts layer because it depends on ``write_routes`` (scripts) + ``SUMOEnv``
(src); the pure-src training loop stays factory-injected and SUMO-free.
"""

from __future__ import annotations

from scripts.build_routes import write_routes
from src.env.sumo_env import SUMOEnv
from src.ml.hybrid_wrapper import HybridStateWrapper
from src.ml.lstm_model import LSTMForecaster
from src.scenarios.config import SCENARIO_DIR, load_scenario


def load_scenario_by_id(scenario_id: str):
    """Load a scenario by id (e.g. ``"SCN-01"`` -> ``config/scenarios/scn_01.yaml``)."""
    return load_scenario(SCENARIO_DIR / f"scn_{scenario_id.split('-')[1]}.yaml")


def build_env(
    scenario_id: str,
    route_seed: int,
    *,
    forecaster: LSTMForecaster | None = None,
    episode_length_s: int | None = None,
    switch_penalty: float = 0.1,
    decision_interval_s: int = 10,
):
    """Build one ``SUMOEnv`` (hybrid-wrapped iff ``forecaster`` given) on a fresh route file.

    ``episode_length_s=None`` uses the scenario's own ``duration_s``. ``sumo_seed`` is the
    ``route_seed`` so traffic and SUMO RNG share one deterministic knob per (scenario, seed).
    """
    scenario = load_scenario_by_id(scenario_id)
    env = SUMOEnv(
        write_routes(scenario, route_seed),
        episode_length_s=episode_length_s or scenario.duration_s,
        decision_interval_s=decision_interval_s,
        switch_penalty=switch_penalty,
        sumo_seed=route_seed,
        signal_mode="rl",
    )
    return HybridStateWrapper(env, forecaster) if forecaster is not None else env
