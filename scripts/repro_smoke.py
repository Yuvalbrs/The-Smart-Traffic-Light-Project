"""T-01-07 - Weekly reproducibility smoke test.

Re-runs the fixed reference episode and compares its JSONL hash to the committed
golden (``golden_hashes.json``, written by T-01-10). Prints PASS/FAIL and exits 0
(match) or 1 (mismatch / missing golden) so it can run as a Friday cron job or a CI
gate. A mismatch means a result-affecting change crept in (SUMO version, net, route
generator, env stepping, or tracer) - investigate before trusting new runs.

Run::

    python -m scripts.repro_smoke
"""

from __future__ import annotations

import json
import sys

from src.repro.reference import GOLDEN_FILE, compute_reference_hash


def main() -> None:
    if not GOLDEN_FILE.exists():
        print(f"[repro] FAIL - no golden file ({GOLDEN_FILE.name}); run scripts.golden_hash first")
        sys.exit(1)

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    current = compute_reference_hash()
    match = current["sha256"] == golden["sha256"]

    print(f"[repro] reference: {current['scenario']} seed {current['seed']} "
          f"/ {current['episode_length_s']}s ({current['n_frames']} frames)")
    print(f"[repro] golden : {golden['sha256']}  ({golden.get('sumo_version', '?')})")
    print(f"[repro] current: {current['sha256']}  ({current['sumo_version']})")
    if match:
        print("[repro] PASS - simulation reproduces the golden hash")
        sys.exit(0)
    print("[repro] FAIL - hash drift; a result-affecting change crept in. Investigate "
          "before trusting new runs (or re-bless via scripts.golden_hash if intended).")
    sys.exit(1)


if __name__ == "__main__":
    main()
