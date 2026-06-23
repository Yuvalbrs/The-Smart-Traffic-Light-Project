"""T-01-10 - Create / refresh the committed golden hash of the reference episode.

One-time bootstrap (re-run deliberately when the SUMO version or the pipeline is
intentionally changed): run the fixed reference episode (src/repro/reference.py),
hash its JSONL trace, and write ``golden_hashes.json``. The weekly smoke test
(T-01-07, scripts/repro_smoke.py) reads that file and fails loudly if a future run
no longer reproduces the hash.

Run::

    python -m scripts.golden_hash
"""

from __future__ import annotations

import json
import sys

from src.repro.reference import GOLDEN_FILE, compute_reference_hash


def main() -> None:
    record = compute_reference_hash()
    GOLDEN_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("[golden] reference episode: "
          f"{record['scenario']} seed {record['seed']} / {record['episode_length_s']}s "
          f"({record['n_frames']} frames, {record['sumo_version']})")
    print(f"[golden] sha256 = {record['sha256']}")
    print(f"[golden] wrote -> {GOLDEN_FILE.name}  (commit this file)")
    sys.exit(0)


if __name__ == "__main__":
    main()
