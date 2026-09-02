"""The movement spec must ship with the repo, and must not drift from the vault original.

`build_net()` runs on every live-session start, and it reads this spec. It used to read it from an
absolute path inside the author's Obsidian vault, so on any other machine - a clone, a marker's
laptop, CI - every episode failed with FileNotFoundError before it started, for every controller,
including the baselines that need no checkpoint at all. Nothing caught it because the tests that
touch the spec were decorated `skipif(not path.exists())`: on the only machine where it was
missing, they simply did not run.

The spec now lives at `config/movements.yaml` and the vault keeps the authoring copy. That trade
has a new failure mode - two copies that disagree - so it is tested here rather than hoped for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.env.intersection import MOVEMENTS_SPEC

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The authoring copy. Absent on every machine but the author's, which is the whole point.
_VAULT_COPY = Path(
    r"C:\Year3\Obsidian\Yuval\30_Projects\smart-traffic-rl\specs\movements.yaml"
)


def test_the_spec_ships_inside_the_repo() -> None:
    """A clone must carry it. This is the test that would have caught the original bug."""
    assert MOVEMENTS_SPEC.exists(), f"{MOVEMENTS_SPEC} is missing - a clone cannot run"
    assert MOVEMENTS_SPEC.is_relative_to(_REPO_ROOT), (
        f"the runtime spec resolves OUTSIDE the repo ({MOVEMENTS_SPEC}); a clone would not have it"
    )


def test_no_source_file_points_at_an_absolute_authoring_path() -> None:
    """Guards the regression directly: a hard-coded C:\\...\\Obsidian path in the runtime.

    Scans the shipped source rather than asserting on one constant, because the original bug was
    duplicated - `src/env/intersection.py` and `scripts/build_network.py` each had their own copy
    of the literal, and fixing one would have left the other.
    """
    # An absolute Windows path literal, not the word "Obsidian" - the fix's own comments explain
    # what went wrong and naming it there is correct. What must never come back is a drive-letter
    # path the runtime opens.
    absolute_path = re.compile(r'r?"[A-Za-z]:\\[^"]*\.yaml"')
    offenders = []
    for folder in ("src", "scripts"):
        for path in (_REPO_ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if absolute_path.search(text):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"these shipped files reference the author's vault by absolute path: {offenders}. "
        f"Anything the runtime reads has to live in the repo."
    )


def test_the_spec_is_well_formed_and_complete() -> None:
    """Twelve movements and eight phases - the shapes the observation and action spaces assume."""
    spec = yaml.safe_load(MOVEMENTS_SPEC.read_text(encoding="utf-8"))
    assert spec["geometry"]["approaches"] == ["N", "E", "S", "W"]
    # Both are mappings keyed by id: movements by "M0".."M11", phases by int 0..7.
    assert len(spec["movements"]) == 12, "obs_dim 20 = 12 movements + 8 phases"
    assert len(spec["phases"]) == 8, "action_space is Discrete(8)"
    assert sorted(spec["movements"]) == sorted(f"M{i}" for i in range(12))
    assert sorted(spec["phases"]) == list(range(8))
    for name, movement in spec["movements"].items():
        assert movement["approach"] in ("N", "E", "S", "W"), name
        assert movement["turn"] in ("left", "through", "right"), name


@pytest.mark.skipif(not _VAULT_COPY.exists(), reason="authoring vault not on this machine")
def test_the_shipped_copy_matches_the_authoring_copy() -> None:
    """Drift guard - only meaningful where both copies exist, i.e. the author's machine.

    Compares parsed YAML, not bytes: a comment or a line ending is allowed to differ, the spec
    itself is not. Fails loudly rather than skipping, because a silent skip is what let the
    original bug live.
    """
    shipped = yaml.safe_load(MOVEMENTS_SPEC.read_text(encoding="utf-8"))
    authored = yaml.safe_load(_VAULT_COPY.read_text(encoding="utf-8"))
    assert shipped == authored, (
        "config/movements.yaml has drifted from the vault original. Re-copy it: the vault is "
        "where it is authored, the repo copy is what actually runs."
    )
