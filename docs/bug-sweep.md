# Bug sweep — 5 screens × 4 states

Run 2026-09-02, before the presentation. The point of a table is that every cell gets *looked
at*: clicking around until nothing obvious breaks finds the bugs you happen to walk past, and
the states nobody visits on purpose — the hub down, a scenario with no data — are exactly the
ones that embarrass you in front of an audience.

Method: the hub (`uvicorn src.api.server:app`) serving the built SPA at `127.0.0.1:8000`, driven
in a real browser. The "no server" column was produced by killing the hub with the page already
open, which is the realistic failure (in the built app the hub serves the page, so a hub that was
never up means no page at all).

**Seven defects found, all fixed.** Five were found by reading the code before running anything;
two more — both in the Compare tab, both invisible to the tests — only showed up on screen.

## React SPA

| Screen | no server | server up, no simulation | simulation running | simulation finished |
|---|---|---|---|---|
| **Live** | "cannot reach the hub — TypeError: Failed to fetch" + **retry** button; `ws/dashboard: closed` badge red; Replay browser "0 runs recorded" + error **[was: blank, silent panel — fixed]** | form renders, all 11 scenarios incl. SCN-R1; "no live session"; "no live frame yet"; "no data yet" **[last two were empty axis-only charts — fixed]** | phase P0/P1 + group + sim time; queue/pressure bars live; 3 KPI lines advancing; `state: running`, frame count rising | `state: finished`, frames 60; charts hold final values; start re-enabled, stop disabled. `stopped` verified separately and reads `state: stopped` |
| **Train** | "cannot reach the hub — TypeError: Failed to fetch"; models list shows its own error **[was: "no training job started yet" — a lie when the server is down; fixed]** | "no training job started yet" (correct here — the hub answers 404); models table populated, user model badged `yours` | progress bar + live reward/epsilon curve over `/ws/training` | `status: done`, 2/2 episodes (100%), curve retained, `run_dir` shown |
| **Compare** | error shown; scenario picker falls back to its built-in list and stays usable | table + both bar charts render **[bars were invisible — fixed]**; honest caveats and the pre-registered negative result printed | (independent of the live session — reads the database) | user-trained rows shown but excluded from winning a column |

Scenario with **no data** (SCN-08, a 404 from `/comparison`): message only.
**[was: the message printed above SCN-05's table and charts, with nothing on the numbers saying
which scenario they belonged to — fixed]**

## Unity client

| Screen | no server | server up, no simulation | simulation running | simulation finished |
|---|---|---|---|---|
| **Live 3-D view** | "connecting (err)" from `SumoSocket.LastError` | "connected — no session running" | "connected — streaming" | "connected — episode finished (no more frames)" **[was: still read "connected" while the picture froze — indistinguishable from a dead feed; fixed]** |
| **Dashboard** | "Not connected to the hub…" | "Connected. Waiting for a frame…" | tiles/phase/bars update | "Episode finished — values below are the final ones, not a stalled feed" **[was: silently frozen tiles; fixed]** |

**Honest limit on the two Unity rows.** The code paths were read, the project compiles clean
(`Assembly-CSharp.dll` rebuilt, 0 `error CS`), the standalone build succeeded (`errors=0`), and
the built `.exe` was launched and ran with a **1,375-byte player log and 0 exceptions** — the
regression that mattered, since this build once produced 185 MB of NullReferenceExceptions per
hour. The four state *strings* above were **not** visually confirmed on screen, because doing so
means taking over the machine's foreground while it is in use. They are single `switch` arms over
`SessionControl.State`, which the picker already displays correctly, so the risk is low — but
"low risk" is not "seen", and this row is the one to re-check during the rehearsal.

## The seven defects

Found by reading the code:

1. `ControlPanel.tsx:22-29` — `getControllers()` had no `.catch`; a rejected call left `options`
   null forever and the entire form is behind `{options && …}`. Hub down ⇒ blank, silent panel.
2. `TrainingPanel.tsx:82-84,163-165` — `.catch(() => {})` swallowed every error, so "server down"
   and "no job yet" rendered identically.
3. `KpiCharts.tsx:16-21` — an empty series still draws axes; idle looked like a running session
   pinned at zero.
4. Scenario list existed in **three** hand-synced copies and had drifted: hub 11, Unity 10
   (missing SCN-R1, so measured demand was unreachable from the 3-D client), React 5. React now
   reads `GET /controllers`; Unity keeps its copy behind a drift test
   (`tests/test_client_scenario_lists.py`, mutation-tested).
5. Unity had no "episode finished" state.

Found only by running it:

6. `MovementBars.tsx` — same empty-state defect as (3): `?? 0` coerces a missing frame to a full
   set of zero bars.
7. **`ComparisonView.tsx` — the two bar charts rendered no bars at all.** `<Bar>` was missing
   `isAnimationActive={false}`; under recharts 3 inside a `ResponsiveContainer` that sizes after
   first paint, the entry animation never runs and the layer emits **zero rects**. Axes,
   gridlines and category labels all present, plot area empty — and the y-axis still scaled to
   the real data, so it looked deliberate. `MovementBars` already carried this flag with a
   comment calling it "load-bearing"; the comparison charts never got it. These two charts are
   the "why ours is best" visual, and they were blank.
8. **`ComparisonView.tsx` — a failed fetch left the previous scenario's data on screen.** Picking
   a scenario with no episodes printed the error *above* the old table and charts. One scenario's
   results read as another's.

(Seven distinct defects; 6–8 are numbered separately above because 3 and 6 share a root cause.)

## Known cosmetic issue, not fixed

The models table can show `3/2` in the episodes column — `episodes_trained` from an earlier run
against `episodes` requested by a later one, when a user model is retrained at the same
variant+seed. Display only; no result depends on it. Left alone this close to the deadline.

## Re-run

```
LIBSUMO_AS_TRACI=1 .venv/Scripts/python -m uvicorn src.api.server:app --port 8000
# open http://127.0.0.1:8000/ , then for the "no server" column kill the hub with the page open
```
