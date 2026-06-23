"""T-01-10 / T-01-07 - the fixed reference episode and its golden hash.

A single deterministic episode (fixed scenario + seed + length, Webster control)
run with JSONL tracing on. SUMO is deterministic given the same binary version,
net, route, seed, and step settings (research-sumo.md s5), and the JsonlWriter
emits byte-identical output (sorted keys, compact, fixed newline) - so the SHA-256
of the trace is a stable fingerprint of the whole simulation pipeline.

A changed hash means something that affects results changed: the SUMO version, the
net, the route generator, the env stepping, or the tracer. That is exactly what the
weekly smoke test (T-01-07) is meant to catch before it silently corrupts a run.

The reference is intentionally short (a few hundred frames) so the weekly check is
seconds, not minutes.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.env.sumo_env import SUMOEnv
from src.provenance.versions import sumo_version
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = _REPO_ROOT / "golden_hashes.json"  # committed reference (T-01-10)

# The fixed reference episode (heavy scenario -> never empties early -> stable
# frame count; one seed; short horizon for a fast weekly check).
REFERENCE_SCENARIO = "SCN-02"
REFERENCE_SEED = 0
REFERENCE_LENGTH_S = 300


def compute_reference_hash() -> dict[str, Any]:
    """Run the reference episode with tracing and return its hash + provenance.

    Returns
    -------
    dict
        ``{"sha256", "n_frames", "sumo_version", "scenario", "seed",
        "episode_length_s"}`` - identical inputs reproduce ``sha256``.
    """
    build_net()
    scn = load_scenario(SCENARIO_DIR / f"scn_{REFERENCE_SCENARIO.split('-')[1]}.yaml")
    route = write_routes(scn, REFERENCE_SEED)
    ctrl = WebsterController(webster_plan_for_scenario(scn))

    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "reference.jsonl"
        env = SUMOEnv(
            route,
            episode_length_s=REFERENCE_LENGTH_S,
            sumo_seed=REFERENCE_SEED,
            trace_path=trace,
        )
        try:
            obs, info = env.reset()
            ctrl.reset(env)
            done = False
            while not done:
                action = ctrl.select_action(obs, info["mask"])
                obs, _r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        finally:
            env.close()
        data = trace.read_bytes()

    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_frames": data.count(b"\n"),
        "sumo_version": sumo_version() or "unknown",
        "scenario": REFERENCE_SCENARIO,
        "seed": REFERENCE_SEED,
        "episode_length_s": REFERENCE_LENGTH_S,
    }
