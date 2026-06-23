"""Non-RL baseline controllers (baselines-implementation.md).

Three locked baselines span the difficulty spectrum, all driving the same
:class:`src.env.sumo_env.SUMOEnv` for an apples-to-apples comparison:

- :mod:`src.baselines.webster` - Webster's classical fixed-time controller (floor).
- :mod:`src.baselines.max_pressure` - greedy max-pressure (strong rule-based).
- SUMO actuated (T-02-06) - to come.
"""

from __future__ import annotations

from src.baselines.max_pressure import MaxPressureController
from src.baselines.webster import (
    WebsterController,
    WebsterPlan,
    compute_webster_plan,
    webster_plan_for_scenario,
)

__all__ = [
    "MaxPressureController",
    "WebsterController",
    "WebsterPlan",
    "compute_webster_plan",
    "webster_plan_for_scenario",
]
