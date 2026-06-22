"""T-01-06 - Provenance version strings + filename helpers (hard-rule #7).

The chain ``data_version -> lstm_version -> run_id`` (data-schema.md s6,
research-eval-repro.md s3):

* ``data_version``  - deterministic hash of (scenario configs, generator code
  git-sha, generation seed, SUMO version). Same inputs -> same string, always.
* ``lstm_version``  - deterministic hash of (data_version, lstm config, training
  code git-sha, training seed).
* ``run_id``        - a UUID per DQN run. GPU training is **not** bit-reproducible,
  so the run_id does not hash its inputs; it is a unique tag and the inputs are
  *recorded* alongside it (in the SQLite ``experiment_run`` row).

The hash functions are pure (all inputs explicit) so they are trivially testable
and stable; the environment collectors (``git_sha``, ``sumo_version``,
``torch_versions``) are separate, best-effort, and never raise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Length of the hex digest kept in a version string. 12 hex chars = 48 bits,
#: ample to avoid collisions across this project's artifact count.
_HASH_LEN = 12


def _sha256_hex(blob: bytes) -> str:
    """Return the full hex SHA-256 of ``blob``."""
    return hashlib.sha256(blob).hexdigest()


def _canonical(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical (sorted, compact) JSON bytes for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable short hash of a config dict (order-independent)."""
    return _sha256_hex(_canonical(config))[:_HASH_LEN]


def hash_files(paths: list[str | Path]) -> str:
    """Return a stable short hash over the contents of ``paths``.

    Files are hashed in sorted-name order so the result is independent of the
    order they are passed (used to fingerprint the scenario configs).
    """
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(Path(p).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:_HASH_LEN]


def data_version(
    *,
    scenario_configs_hash: str,
    generator_git_sha: str,
    generation_seed: int,
    sumo_version: str,
) -> str:
    """Return the deterministic ``data_version`` string.

    Parameters
    ----------
    scenario_configs_hash : str
        Hash of the scenario YAMLs (e.g. from :func:`hash_files`).
    generator_git_sha : str
        Code commit of the generator at generation time.
    generation_seed : int
        Seed used to generate the dataset.
    sumo_version : str
        Pinned SUMO version string.

    Returns
    -------
    str
        ``"data-<12 hex>"`` - identical inputs always yield the same value.
    """
    digest = _sha256_hex(
        _canonical(
            {
                "scenario_configs_hash": scenario_configs_hash,
                "generator_git_sha": generator_git_sha,
                "generation_seed": generation_seed,
                "sumo_version": sumo_version,
            }
        )
    )[:_HASH_LEN]
    return f"data-{digest}"


def lstm_version(
    *,
    data_version: str,
    lstm_config_hash: str,
    training_code_git_sha: str,
    training_seed: int,
) -> str:
    """Return the deterministic ``lstm_version`` string (depends on data_version)."""
    digest = _sha256_hex(
        _canonical(
            {
                "data_version": data_version,
                "lstm_config_hash": lstm_config_hash,
                "training_code_git_sha": training_code_git_sha,
                "training_seed": training_seed,
            }
        )
    )[:_HASH_LEN]
    return f"lstm-{digest}"


def new_run_id() -> str:
    """Return a fresh UUID4 string for a DQN run (inputs are recorded, not hashed)."""
    return str(uuid.uuid4())


def checkpoint_filename(
    kind: str,
    *,
    data_version: str,
    lstm_version: str | None = None,
    step: int | None = None,
    ext: str = ".pt",
) -> str:
    """Build a checkpoint filename that embeds the version chain (data-schema.md s6).

    Examples
    --------
    ``dqn__data-ab12cd34ef56__lstm-99887766aabb__step50000.pt``
    ``lstm__data-ab12cd34ef56__step20000.pt``
    """
    parts = [kind, data_version]
    if lstm_version is not None:
        parts.append(lstm_version)
    if step is not None:
        parts.append(f"step{step}")
    return "__".join(parts) + ext


# --- best-effort environment collectors (never raise) ---


def git_sha(repo_root: str | Path = _REPO_ROOT, *, short: bool = False) -> str | None:
    """Return the current git commit of ``repo_root``, or ``None`` if unavailable."""
    cmd = ["git", "-C", str(repo_root), "rev-parse"]
    cmd += ["--short", "HEAD"] if short else ["HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sumo_version() -> str | None:
    """Return the installed SUMO version string, or ``None`` if unavailable."""
    try:
        from sumolib import checkBinary

        out = subprocess.run(
            [checkBinary("sumo"), "--version"], capture_output=True, text=True, check=True
        )
        for line in out.stdout.splitlines():
            if line.strip().startswith("SUMO"):
                return line.strip()
        return out.stdout.splitlines()[0].strip() if out.stdout else None
    except Exception:  # noqa: BLE001 - best-effort provenance, must not break a run
        return None


def torch_versions() -> dict[str, Any]:
    """Return torch/CUDA provenance for a run_id (all ``None`` if torch absent).

    Records the GPU stack so the run_id's documented non-reproducibility caveat is
    auditable (research-eval-repro.md s3).
    """
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": None, "cudnn_deterministic": None}
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
