"""The two clients hand-copy the hub's scenario and controller lists. Alarm on the drift.

The React SPA now fetches both lists from ``GET /controllers`` at mount, so it cannot drift. The
Unity client still hard-codes them in ``RunSetupUI.cs``: fetching them there changes the startup
path of the screen the whole demo runs through, which is not a change worth making hours before a
presentation.

A hand-synced list with no alarm on it drifts silently, and this one already did - ``SCN-R1``
shipped in the hub and never reached Unity, so the measured-demand scenario was simply unreachable
from the 3-D client and nobody noticed. These tests fail the moment the copies disagree again.

Note on the parsing: the regexes below assert they matched *something* before comparing. A regex
that silently matches nothing turns "the lists disagree" into a passing test, which is the exact
failure mode this file exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.api.live import CONTROLLERS
from src.api.server import SCENARIOS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_SETUP_UI = _REPO_ROOT / "unity" / "SmartTrafficViz" / "Assets" / "Scripts" / "RunSetupUI.cs"

#: Arterial scenarios need the two-junction network the hub does not build, so they are absent
#: from both the hub and the client on purpose.
_ARTERIAL = ("SCN-A1", "SCN-A2", "SCN-A3", "SCN-A4")


def _source() -> str:
    if not _RUN_SETUP_UI.exists():  # pragma: no cover - only on a checkout without the Unity client
        pytest.skip(f"Unity client not present at {_RUN_SETUP_UI}")
    return _RUN_SETUP_UI.read_text(encoding="utf-8")


def _unity_scene_ids() -> list[str]:
    """Scenario ids from ``RunSetupUI.Scenes``, in declaration order."""
    block = re.search(r"Scene\[\]\s+Scenes\s*=\s*\{(.*?)\n\s*\};", _source(), re.S)
    assert block is not None, "could not find the Scenes array in RunSetupUI.cs"
    ids = re.findall(r'new Scene\(\s*"([^"]+)"', block.group(1))
    assert ids, "matched the Scenes array but extracted no scenario ids - the regex is stale"
    return ids


def _unity_controllers() -> list[str]:
    block = re.search(r"string\[\]\s+Controllers\s*=\s*\{(.*?)\n\s*\};", _source(), re.S)
    assert block is not None, "could not find the Controllers array in RunSetupUI.cs"
    names = re.findall(r'"([^"]+)"', block.group(1))
    assert names, "matched the Controllers array but extracted no names - the regex is stale"
    return names


def test_unity_offers_every_scenario_the_hub_accepts() -> None:
    """A scenario the hub accepts but Unity never lists is unreachable from the 3-D client."""
    expected = [s for s in SCENARIOS if s not in _ARTERIAL]
    assert _unity_scene_ids() == expected


def test_unity_offers_no_scenario_the_hub_would_reject() -> None:
    """The mirror of the above: a stale id here 422s at RUN, after the user has chosen it."""
    assert set(_unity_scene_ids()) <= set(SCENARIOS)


def test_unity_controller_list_matches_the_hub() -> None:
    assert _unity_controllers() == list(CONTROLLERS)


def test_arterial_scenarios_reach_neither_the_hub_nor_the_client() -> None:
    """They need the two-junction network; offering them anywhere would be a dead end."""
    assert not set(_ARTERIAL) & set(SCENARIOS)
    assert not set(_ARTERIAL) & set(_unity_scene_ids())
