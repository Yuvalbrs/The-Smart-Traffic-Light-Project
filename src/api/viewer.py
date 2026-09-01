"""Launch and track the Unity 3D viewer, so the dashboard and the visualisation are one app.

The viewer is a separate OS window rather than a panel inside the dashboard, and that is a
constraint rather than a preference: rendering Unity inside the browser needs a WebGL player, the
WebGL module is not installed on this machine, and ``SumoSocket`` uses
``System.Net.WebSockets.ClientWebSocket``, which has no WebGL implementation at all - WebGL builds
have no threads or sockets and need a JavaScript bridge instead. Both are real pieces of work, not
a switch.

What makes the two halves one application is not that they share a window; it is that they share a
**feed**. The viewer consumes ``/ws/unity`` and the dashboard consumes ``/ws/dashboard``, and both
are fan-outs of the same 1 Hz frames from the same running episode, so the picture and the numbers
cannot disagree. This module adds the last missing piece: starting the viewer from the application
itself, instead of asking the user to go and open the Unity Editor.

If the standalone player has not been built, every endpoint here says so plainly and names the
command that builds it, rather than failing in a way that looks like a bug.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where ``BuildScript.BuildWindows`` writes the player. Fixed, never taken from a request: this
#: module starts a process, and a path that came from an HTTP body would be a way to start any of
#: them.
VIEWER_EXE = _REPO_ROOT / "unity" / "SmartTrafficViz" / "Build" / "SmartTrafficViz.exe"

BUILD_HINT = (
    'Unity.exe -quit -batchmode -nographics -projectPath unity/SmartTrafficViz '
    "-executeMethod SmartTrafficViz.EditorTools.BuildScript.BuildWindows "
    "-logFile unity/build_standalone.log   (close the Unity Editor first - it locks the project)"
)

EDITOR_HINT = (
    "Alternatively open unity/SmartTrafficViz in Unity 6000.0.41f1 and press Play; it connects to "
    "the same /ws/unity feed and looks identical on screen."
)


class ViewerManager:
    """Owns at most one viewer process, and knows whether one can be started at all."""

    def __init__(self, exe: Path | None = None) -> None:
        self._exe = exe or VIEWER_EXE
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def available(self) -> bool:
        """Whether a built player exists to launch."""
        return self._exe.is_file()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        """What the UI needs to decide between a button and an explanation."""
        return {
            "available": self.available,
            "running": self.running,
            "path": str(self._exe),
            "hint": None if self.available else BUILD_HINT,
            "editor_hint": EDITOR_HINT,
        }

    def start(self) -> dict[str, Any]:
        """Launch the viewer. Idempotent: a second call while it runs is not an error."""
        if not self.available:
            raise FileNotFoundError(
                f"the Unity viewer has not been built ({self._exe} does not exist). {BUILD_HINT}"
            )
        if self.running:
            return self.status()
        # Detached, with its own working directory, so the viewer keeps running if the hub is
        # restarted and its stdout never fills a pipe nobody drains.
        self._proc = subprocess.Popen(
            [str(self._exe)],
            cwd=str(self._exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.status()

    def stop(self) -> bool:
        """Close the viewer. ``False`` when it was not running."""
        if not self.running:
            return False
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - the window is not cooperating
            self._proc.kill()
        return True
