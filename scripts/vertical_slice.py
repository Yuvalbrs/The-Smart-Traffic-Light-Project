"""T-00-01 - Trivial end-to-end vertical slice.

Proves the SUMO <-> TraCI <-> Python plumbing end to end: launch SUMO on the
real 4-way intersection (T-01-02) with a generated route file (T-01-08),
connect via TraCI, drive one full 3600 s episode with random ``Discrete(8)``
actions, and exit cleanly with vehicles having actually departed and arrived.

This is deliberately minimal - no RL, no LSTM, no reward, no action masking, no
dashboard, no Unity. It runs on ``config/network/intersection.net.xml`` (the
real net) and a ``config/routes/scn_*.rou.xml`` route file, catching
Python/SUMO version, TraCI connection, network-file and route-loading problems
while there is nothing else to disentangle them from.

The route files are gitignored (regenerable); if the requested one is missing,
generate them first with ``python -m scripts.build_routes``.

Run::

    python -m scripts.vertical_slice                       # scn 01, seed 0
    python -m scripts.vertical_slice --scenario 3 --seed 2
    python -m scripts.vertical_slice --gui
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NET_FILE = _REPO_ROOT / "config" / "network" / "intersection.net.xml"
_ROUTES_DIR = _REPO_ROOT / "config" / "routes"
DEFAULT_SCENARIO: int = 1
DEFAULT_SEED: int = 0


def route_file_for(scenario: int, seed: int) -> Path:
    """Return the path to the generated route file for ``(scenario, seed)``.

    Mirrors the naming written by ``scripts.build_routes`` (zero-padded to two
    digits, e.g. ``scn_01_seed_00.rou.xml``). The file may not exist; callers
    are responsible for checking.

    Parameters
    ----------
    scenario : int
        Scenario number (1-5), matching ``config/scenarios/scn_NN.yaml``.
    seed : int
        Generation seed for the route file.

    Returns
    -------
    Path
        Absolute path to the corresponding ``.rou.xml`` under ``config/routes``.
    """
    return _ROUTES_DIR / f"scn_{scenario:02d}_seed_{seed:02d}.rou.xml"


def run_episode(route_file: Path, use_gui: bool = False) -> int:
    """Run one full vertical-slice episode end to end.

    Launches SUMO on the real net (``_NET_FILE``) with ``route_file``, connects
    via TraCI, and drives a random phase every ``DECISION_INTERVAL_S`` seconds
    until the episode horizon is reached or no vehicles remain. The TraCI
    connection is always closed.

    Parameters
    ----------
    route_file : Path
        The ``.rou.xml`` route file to load against the real network.
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
        If the network exposes no traffic light to control.
    """
    sumo_binary = checkBinary("sumo-gui" if use_gui else "sumo")
    cmd = [
        sumo_binary,
        "--net-file",
        str(_NET_FILE),
        "--route-files",
        str(route_file),
        "--step-length",
        str(STEP_LENGTH_S),
        "--seed",
        str(SEED),
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
            raise RuntimeError("network has no traffic light to control")
        tls_id = tls_ids[0]
        n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        print(f"[slice] controlling TLS {tls_id!r} with {n_phases} phases")

        rng = random.Random(SEED)
        step = 0
        departed_total = 0
        arrived_total = 0
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
                    departed_total += traci.simulation.getDepartedNumber()
                    arrived_total += traci.simulation.getArrivedNumber()
                    step += 1
        except traci.exceptions.FatalTraCIError:
            # Closing the GUI window mid-run drops the connection; in GUI mode
            # that is a normal way to stop watching, not a failure.
            if not use_gui:
                raise
            print(f"[slice] GUI window closed - stopped early at step {step}")
            return step

        sim_time = traci.simulation.getTime()
        print(
            f"[slice] episode complete: {step} steps, sim_time={sim_time:.0f}s, "
            f"departed={departed_total}, arrived={arrived_total}"
        )
        return step
    finally:
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    """Entry point: parse args, run one episode, and exit 0 on success."""
    parser = argparse.ArgumentParser(
        description="Vertical slice: random Discrete(8) policy on the real net.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=DEFAULT_SCENARIO,
        help="scenario number 1-5 (config/scenarios/scn_NN.yaml); default 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="route-file generation seed; default 0",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="watch the run in the SUMO GUI instead of running headless",
    )
    args = parser.parse_args()

    route_file = route_file_for(args.scenario, args.seed)
    if not route_file.exists():
        print(
            f"[slice] route file not found: {route_file}\n"
            f"[slice] route files are gitignored (regenerable); generate them with:\n"
            f"[slice]     python -m scripts.build_routes",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[slice] net={_NET_FILE.name}  routes={route_file.name}")

    steps = run_episode(route_file, use_gui=args.gui)
    print(f"[slice] OK - exited cleanly after {steps} steps")
    sys.exit(0)


if __name__ == "__main__":
    main()
