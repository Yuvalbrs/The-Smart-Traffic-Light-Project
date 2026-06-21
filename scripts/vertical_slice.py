"""T-00-01 - Trivial end-to-end vertical slice.

Proves the SUMO <-> TraCI <-> Python plumbing end to end: launch SUMO on a
throwaway stub network, connect via TraCI, drive one full 3600 s episode with
random ``Discrete(8)`` actions, and exit cleanly.

This is deliberately minimal - no RL, no LSTM, no reward, no action masking, no
dashboard, no Unity. It runs on ``scripts/_stub/`` (a throwaway 4-arm cross),
NOT the real network (T-01-02). Once the real net exists the slice is
re-pointed at it; until then this catches Python/SUMO version, TraCI
connection, network-file and environment-setup problems while there is nothing
else to disentangle them from.

Run::

    python -m scripts.vertical_slice
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import traci
from sumolib import checkBinary

# Episode constants mirror the locked sim config (vault decisions.md): 1.0 s
# step length, a decision every 10 simulated seconds, a one-hour episode, and
# the Discrete(8) NEMA action space. The stub TLS may expose fewer phases, so
# actions are mapped onto it modulo the available phase count.
STEP_LENGTH_S: float = 1.0
DECISION_INTERVAL_S: int = 10
EPISODE_LENGTH_S: int = 3600
ACTION_SPACE_N: int = 8
SEED: int = 42

_STUB_DIR = Path(__file__).resolve().parent / "_stub"
_SUMOCFG = _STUB_DIR / "stub.sumocfg"


def run_episode(use_gui: bool = False) -> int:
    """Run one full vertical-slice episode end to end.

    Launches SUMO on the stub config, connects via TraCI, and drives a random
    phase every ``DECISION_INTERVAL_S`` seconds until the episode horizon is
    reached or no vehicles remain. The TraCI connection is always closed.

    Parameters
    ----------
    use_gui : bool, optional
        If ``True``, launch the ``sumo-gui`` visual front-end (auto-started,
        with a small per-step delay so vehicles are watchable) instead of the
        headless ``sumo`` binary. Default ``False``.

    Returns
    -------
    int
        The number of simulation steps actually executed.

    Raises
    ------
    RuntimeError
        If the stub network exposes no traffic light to control.
    """
    sumo_binary = checkBinary("sumo-gui" if use_gui else "sumo")
    cmd = [
        sumo_binary,
        "-c",
        str(_SUMOCFG),
        "--step-length",
        str(STEP_LENGTH_S),
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
    ]
    if use_gui:
        # Auto-run on launch, slow each step to ~150 ms so it is watchable, and
        # close the window when the episode ends. Drag the GUI "Delay" slider to
        # change speed live.
        cmd += ["--start", "--delay", "150", "--quit-on-end"]
    traci.start(cmd)
    try:
        print(f"[slice] connected to SUMO {traci.getVersion()[1]}")

        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("stub network has no traffic light to control")
        tls_id = tls_ids[0]
        n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        print(f"[slice] controlling TLS {tls_id!r} with {n_phases} phases")

        rng = random.Random(SEED)
        step = 0
        # getMinExpectedNumber() == 0 -> no vehicles loaded or in transit; this
        # is the natural early-termination guard (the same pattern T-02-01 bakes
        # in as a B3 guard for real episodes).
        try:
            while step < EPISODE_LENGTH_S and traci.simulation.getMinExpectedNumber() > 0:
                action = rng.randrange(ACTION_SPACE_N)
                traci.trafficlight.setPhase(tls_id, action % n_phases)

                for _ in range(DECISION_INTERVAL_S):
                    if step >= EPISODE_LENGTH_S or traci.simulation.getMinExpectedNumber() == 0:
                        break
                    traci.simulationStep()
                    step += 1
        except traci.exceptions.FatalTraCIError:
            # Closing the GUI window mid-run drops the connection; in GUI mode
            # that is a normal way to stop watching, not a failure.
            if not use_gui:
                raise
            print(f"[slice] GUI window closed - stopped early at step {step}")
            return step

        sim_time = traci.simulation.getTime()
        print(f"[slice] episode complete: {step} steps, sim_time={sim_time:.0f}s")
        return step
    finally:
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    """Entry point: parse args, run one episode, and exit 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="watch the run in the SUMO GUI instead of running headless",
    )
    args = parser.parse_args()

    steps = run_episode(use_gui=args.gui)
    print(f"[slice] OK - exited cleanly after {steps} steps")
    sys.exit(0)


if __name__ == "__main__":
    main()
