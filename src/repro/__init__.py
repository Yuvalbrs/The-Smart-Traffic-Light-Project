"""Reproducibility layer - the fixed reference episode + its golden hash.

``reference`` defines one deterministic fixed-seed episode and hashes its JSONL
trace. ``scripts/golden_hash.py`` (T-01-10) commits that hash; ``scripts/
repro_smoke.py`` (T-01-07) re-runs and compares - both call the same reference
function, so there is no drift between what was frozen and what is checked.
"""

from __future__ import annotations

from src.repro.reference import GOLDEN_FILE, compute_reference_hash

__all__ = ["GOLDEN_FILE", "compute_reference_hash"]
