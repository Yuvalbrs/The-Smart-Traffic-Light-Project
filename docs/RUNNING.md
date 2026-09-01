# Running the app

This is the operator doc: how to start the packaged app and how to run the 3-minute demo.
For the project spec/blueprint, see [`docs/README.md`](README.md) (it lives in the vault).

## Run the app

**Prerequisites** (the launcher checks these itself and tells you exactly what's missing):

- `SUMO_HOME` set and pointing at a real SUMO install (verified locally: SUMO 1.27.0,
  `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`).
- `.venv\` present with dependencies installed (`pip install -r requirements.txt`).
- `data\traffic.db` present (`.venv\Scripts\python.exe -m scripts.init_db` if not).
- Node.js + `npm` on `PATH`, **only** if `frontend\dist\` does not already exist - the launcher
  builds it for you the first time (`npm install` + `npm run build`), which takes a few minutes.
  Already-built or rebuilt, subsequent runs skip straight to starting the server.

**The one command:**

- Double-click **`run_app.bat`** in the repo root, **or**
- From a terminal: `.venv\Scripts\python.exe run_app.py`

**What you should see:**

1. A prerequisite check (instant, or a few minutes the very first time if it has to build the
   frontend - the console prints progress the whole way, it is not hung).
2. `[run_app] waiting for the hub at http://127.0.0.1:8000/health ...` for a couple of seconds
   while uvicorn imports SUMO/libsumo and comes up.
3. A banner:
   ```
   ============================================================
     Smart Traffic Intersection - running at http://127.0.0.1:8000/
     Press Ctrl+C here to stop it.
   ============================================================
   ```
4. Your default browser opens on the dashboard automatically.
5. **Ctrl+C** in that console window stops the server cleanly (no traceback). Closing the
   console window (or the `run_app.bat` window) also stops it.

Useful flags: `--no-browser` (don't auto-open), `--port N` (default 8000, use this if something
else already owns 8000), `--skip-build` (fail fast instead of building if `frontend\dist\` is
missing - useful in CI), `--rebuild` (force a fresh frontend build even if `dist\` exists,
e.g. after pulling frontend changes).

If it fails, the message on screen tells you the exact next command to run (create the venv,
init the DB, set `SUMO_HOME`, free up the port, etc.) - that is by design, so a demo-night
failure is a two-second fix, not a debugging session.

---

## Demo script (3 minutes)

Run through this once beforehand with a stopwatch - the timings assume you talk while things
load, not after. Have `run_app.bat` already running and the browser tab open before you hit
record; do not show the terminal boot-up in the recording.

**0:00 - 0:20 | Open on the Live tab**
Say what this is: a plain-DQN controller (with an optional LSTM traffic forecast) picking the
next NEMA signal phase at a 4-way intersection, simulated in SUMO, benchmarked against three
classic baselines. Point at the tab bar - **Live / Train / Compare** - and say each does one
job: watch it drive, watch it learn, prove it's better.

**0:20 - 1:00 | Live tab - start an episode**
Click **Live**. Pick a controller (default `sel/plain` - the shipped product) and a scenario,
hit **Start**. While it spins up, narrate: this is a real SUMO simulation running headless on
this machine right now, streamed over WebSocket at 1 Hz. Once frames arrive, point at three
things as they move: the vehicles/queues, the current signal phase changing, and the KPI numbers
ticking (queue length, throughput) - all live, all the same episode.

**1:00 - 1:40 | Train tab - show learning happening**
Click **Train**. Pick a model variant (e.g. `hybrid`, the DQN + frozen LSTM forecast), a seed,
a small episode count for the demo. Hit start and let the reward curve begin climbing on
screen. Say what's being optimized (episodic reward = throughput up, queueing/switching down)
and that this is the same training loop that produced the shipped checkpoints, just running live
and short for the camera.

**1:40 - 2:30 | Compare tab - the payoff**
Click **Compare**. Point at the per-scenario table: every controller (the RL agent + Webster
fixed-time + max-pressure + SUMO actuated) evaluated on the *same* seeds, so it's an
apples-to-apples comparison, not cherry-picked runs. Point at the bolded/marked best-value cell
per KPI column and read one or two out loud (e.g. "lowest average delay" / "highest throughput").
This is the one screen that answers "does it actually work" without narration.

**2:30 - 3:00 | Unity 3-D (optional close, if time and Unity is already open)**
Say Unity is a separate 3-D *view* of the exact same live episode, not a separate simulation -
same WebSocket feed (`ws/unity`) as the dashboard, so the picture and the numbers can never
disagree. If it's already running (Unity Hub -> `unity/SmartTrafficViz` -> Play, done *before*
recording - first open resolves packages and takes minutes, don't do that live), just cut to it
and let the intersection run for a few seconds with signals changing colour. Full details,
including why it's built the way it is, are in `unity/README.md`. If you're short on time, skip
this section entirely and end on the Compare table instead - it's the stronger closing shot.

**If something breaks on camera:** stop, fix using the on-screen error message from `run_app.py`
(see "Run the app" above), and re-take that segment. Do not try to debug live in the recording.
