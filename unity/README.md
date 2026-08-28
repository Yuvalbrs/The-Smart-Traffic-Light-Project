# Unity client (T-05-03)

Renders a live SUMO episode in 3D from the **same 1 Hz `ws/unity` feed the dashboard uses**, so
the picture and the numbers can never disagree.

## Run it

1. Start the hub:
   ```
   LIBSUMO_AS_TRACI=1 .venv/Scripts/python -m uvicorn src.api.server:app
   ```
2. Unity Hub -> **Add** -> `unity/SmartTrafficViz` -> open with **6000.0.41f1**.
   First open resolves Newtonsoft.Json and takes a few minutes.
3. Press **Play**.
4. Start an episode - dashboard (http://localhost:5173) or REST:
   ```
   curl -X POST http://127.0.0.1:8000/sessions -H "Content-Type: application/json" \
     -d '{"controller":"webster","scenario":"SCN-04","seed":7000,"episode_length_s":600,"speed":1.0}'
   ```
   Use `"speed": 1.0` (or `5.0`). At the API default of `0` the episode runs ~700x faster than
   real time and is over before it is watchable.

No scene to open and nothing to drag into a hierarchy: `Bootstrap` installs the viewer after
scene load, and `IntersectionScene` builds the roads, the twelve signal heads and the camera at
runtime. The on-screen overlay reports socket state, sim time, phase, vehicle count and dropped
frames.

## Why it looks the way it does

Visual-minimal is the agreed bar for this task - correctness over fidelity (`finish-plan.md`
Phase 4). Cars are boxes and signal heads are spheres; what has to be right is that they are in
the *right place*, showing the *right colour*, at the *right time*.

## Design decisions (deviations recorded on purpose)

| Decision | Why |
|---|---|
| **Unity 6** (6000.0.41f1), not 2022 LTS | 2022 LTS is a multi-GB download; Unity 6 was already installed. Newer LTS, same APIs used here. |
| **`System.Net.WebSockets.ClientWebSocket`**, not NativeWebSocket 2.x | The backlog DoD names NativeWebSocket, which exists to work around WebGL's missing System.Net sockets. This is a desktop demo, so the built-in client removes a git-URL package from the critical path. |
| **Built-in render pipeline**, not URP | `Shader.Find` prefers the URP Lit shader and falls back to Standard, so the project works either way with no pipeline asset to configure. |
| **Procedural scene**, no `.unity` asset | Scene files cannot be meaningfully diffed or reviewed; geometry derived from `config/network/intersection.*` cannot drift from the network SUMO simulates. |

Newtonsoft **is** kept: `signal_colors` is a dictionary, which `JsonUtility` cannot deserialise.

## Coordinates

SUMO is 2-D, x east / y north; Unity is y-up. So SUMO `(x, y)` -> Unity `(x, 0, y)`. SUMO
headings are degrees clockwise from north, which is already Unity's y-Euler convention, so angles
pass through unchanged.

Lane placement follows SUMO's own rules - index 0 is the rightmost lane in the direction of
travel, and lanes sit to the right of that direction - checked against a recorded frame rather
than assumed: on the north approach (heading south) lane `n_t_2` is the leftmost, left-turn-only
lane nearest the centreline at `x = -1.6`, which is exactly where vehicle `v120` was.

## Interpolation

Frames arrive at 1 Hz and are interpolated with the **fixed-endpoint** form the T-05-03 DoD
requires:

```csharp
position = Vector3.Lerp(previousFrame, latestFrame, elapsed / frameInterval);
```

Not `Lerp(current, target, k * Time.deltaTime)`. That ease-out version is framerate-dependent -
it converges differently at 30 fps than at 144 - and never quite arrives. The audit flagged it
(orphan T-U05); `notes/03-simulation.md` 6.5 still shows the wrong form.

`frameInterval` is **measured** between arrivals, not hard-coded to 1 s, because the hub's
`speed` control changes how fast frames actually come.

## Known gaps

- **Replay playback is T-05-04**, not this task: the client follows the live channel only.
- `signal.phase_remaining_s` is not rendered. Baseline controllers do not maintain it - webster
  emits values like `86099.0` - so phase colour comes from `signal_colors` instead.
- No camera controls; the view is a fixed 45-degree overhead.
