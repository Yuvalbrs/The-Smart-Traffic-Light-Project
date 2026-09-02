// The live dashboard, drawn inside the application rather than in a browser beside it.
//
// The React dashboard remains the web client and reads the same channel; this is the same
// information rendered natively so the Unity build is a whole application - menu, 3-D view,
// dashboard, results - instead of one window that needs a second one open to be useful.
//
// Everything here is read from the hub's ws/dashboard frames. Nothing is computed locally: the
// hub already derives queues, pressure and the running KPIs once, and a second implementation on
// this side would be free to disagree with the numbers the results tables are built from.
//
// The KPI values are running ESTIMATES and are labelled as such on screen. The project's actual
// KPIs come from the trip-info extractor after an episode ends; presenting a mid-episode estimate
// as a result is exactly the confusion the wire module warns about.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public class DashboardScreen
    {
        /// <summary>Movement labels M0..M11, in the order the wire sends them.</summary>
        private static readonly string[] MovementNames =
        {
            "N left", "N thru", "N right", "E left", "E thru", "E right",
            "S left", "S thru", "S right", "W left", "W thru", "W right",
        };

        private readonly DashboardFeed _feed;
        private DashboardFrame _latest;

        /// <summary>Latest hub session state, pushed in by AppShell's poll. See the note in
        /// TrafficViz: after an episode ends the socket stays open and the tiles simply stop
        /// changing, which is indistinguishable from a dead feed unless we say so.</summary>
        public string SessionState = "unknown";

        /// <summary>Set by AppShell so the dashboard can drive sessions and read the archive -
        /// the React dashboard controls scenes from this screen, and so does this one now.</summary>
        public HubApi Api;
        public SessionControl Session;
        public RunConfig Config = RunConfig.Default;

        private Vector2 _scroll, _runScroll;

        /// <summary>Height the session+replay block actually consumed last frame.
        ///
        /// It used to be a literal in ContentHeight - 60 when the archive was collapsed - while
        /// DrawReplay drew a heading, a scene row, a controller row and two buttons, well over
        /// twice that. The scroll rect was therefore SHORTER than its own contents, so the last
        /// row was unreachable no matter how far you scrolled. Measuring removes the second copy
        /// of the arithmetic; it self-corrects on the frame after any layout change.</summary>
        private float _replayH = 210f;
        private bool _showReplay;

        public DashboardScreen(string wsUnityUrl)
        {
            // ws://host/ws/unity -> ws://host/ws/dashboard, so the two clients can never be
            // pointed at different hubs by a half-edited setting.
            var url = wsUnityUrl.Replace("/ws/unity", "/ws/dashboard");
            _feed = new DashboardFeed(url);
            _feed.Start();
        }

        public bool Connected => _feed.Connected;

        /// <summary>Drain the feed to the newest frame. Call from Update on the main thread.</summary>
        public void Tick()
        {
            while (_feed.TryDequeue(out var frame)) _latest = frame;
        }

        public void Dispose() => _feed.Dispose();

        public void Draw()
        {
            const float pad = 28f;
            // Was capped at 980 px, which on a 2560-wide screen left the panel a narrow column
            // of tiny text in the middle. It now uses the window it is given.
            var w = Mathf.Min(1560f, UnityEngine.Screen.width - pad * 2f);
            // Stops short of the bottom so the shared Live/Dashboard/Train/Compare bar has room.
            var box = new Rect((UnityEngine.Screen.width - w) / 2f, pad, w,
                UnityEngine.Screen.height - pad - 78f);
            GUI.Box(box, GUIContent.none, UITheme.Panel);

            var x = box.x + 20f;
            var y = box.y + 18f;
            var inner = w - 40f;

            GUI.Label(new Rect(x, y, inner, 52f), "Live dashboard", UITheme.Title);
            y += 62f;

            if (_latest == null)
            {
                GUI.Label(new Rect(x, y, inner, 28f),
                    _feed.Connected
                        ? "Connected. Waiting for a frame - start an episode from Controls."
                        : "Not connected to the hub. Is it running on " + _feed.Url + " ?",
                    UITheme.Hint);
                if (!string.IsNullOrEmpty(_feed.LastError))
                {
                    GUI.Label(new Rect(x, y + 32f, inner, 28f), _feed.LastError, UITheme.Hint);
                }
                return;
            }

            // Everything below scrolls: with the panels the React dashboard has - phase, queues,
            // pressures, forecast, replay - the content is taller than any window it runs in.
            var viewH = box.yMax - y - 104f;
            var view = new Rect(x, y, inner, viewH);
            var content = new Rect(0f, 0f, inner - 22f, ContentHeight(_latest));
            _scroll = GUI.BeginScrollView(view, _scroll, content);

            var cy = 0f;
            var cw = content.width;
            cy = DrawKpiTiles(0f, cy, cw, _latest);
            cy = DrawPhase(0f, cy, cw, _latest);
            cy = DrawMovementBars(0f, cy, cw, "Queue per movement (vehicles)", _latest.QueueLengths);
            cy = DrawMovementBars(0f, cy, cw, "Pressure per movement", _latest.Pressures);
            cy = DrawForecast(0f, cy, cw, _latest);
            var replayTop = cy;
            cy = DrawReplay(0f, cy, cw);
            _replayH = cy - replayTop;
            GUI.EndScrollView();
            y = view.yMax + 6f;

            // Tiles keep their last values after an episode ends, which reads exactly like a dead
            // feed. Name the session state so a stopped dashboard is legible rather than alarming.
            var ended = SessionState == "finished" || SessionState == "stopped" || SessionState == "failed";
            if (ended)
            {
                GUI.Label(new Rect(x, box.yMax - 62f, inner, 24f),
                    "Episode " + SessionState + " - values below are the final ones, not a stalled feed.",
                    UITheme.Hint);
            }

            GUI.Label(new Rect(x, box.yMax - 34f, inner, 24f),
                $"sim t={_latest.SimTime:0}s   frame #{_latest.Seq}   received={_feed.Received}   " +
                $"dropped={_feed.Dropped}", UITheme.Hint);
        }

        private static float DrawKpiTiles(float x, float y, float w, DashboardFrame f)
        {
            const float h = 104f;
            var third = (w - 24f) / 3f;
            DrawTile(new Rect(x, y, third, h), "avg wait so far",
                $"{f.RunningKpis.AvgWaitSoFar:0.0} s");
            DrawTile(new Rect(x + third + 12f, y, third, h), "throughput so far",
                $"{f.RunningKpis.ThroughputSoFar:0} vehicles/h");
            DrawTile(new Rect(x + (third + 12f) * 2f, y, third, h), "queue now",
                $"{f.RunningKpis.CurrentQueueTotal:0} vehicles");
            GUI.Label(new Rect(x, y + h + 6f, w, 22f),
                "Running estimates, not the confirmatory KPIs - those come from trip-info after "
                + "the episode ends.", UITheme.Hint);
            return y + h + 38f;
        }

        private static void DrawTile(Rect r, string label, string value)
        {
            GUI.Box(r, GUIContent.none, UITheme.Panel);
            GUI.Label(new Rect(r.x + 16f, r.y + 12f, r.width - 32f, 24f), label, UITheme.Hint);
            GUI.Label(new Rect(r.x + 16f, r.y + 42f, r.width - 32f, 46f), value, UITheme.Tile);
        }

        private static float DrawPhase(float x, float y, float w, DashboardFrame f)
        {
            GUI.Label(new Rect(x, y, w, UITheme.LineH(UITheme.Heading)), "Signal phase", UITheme.Heading);
            y += 34f;
            const float cell = 48f, gap = 9f;
            for (var i = 0; i < 8; i++)
            {
                var r = new Rect(x + i * (cell + gap), y, cell, cell);
                var on = i == f.CurrentPhase;
                var prev = GUI.color;
                // The phase the lights actually show, which during a yellow/all-red transition is
                // still the OUTGOING phase - last_action is where the agent has already decided to
                // go. Showing only one of the two makes a transition look like a glitch.
                GUI.color = on ? UITheme.Accent : new Color(1f, 1f, 1f, 0.18f);
                GUI.Box(r, GUIContent.none, UITheme.Panel);
                GUI.color = prev;
                GUI.Label(r, i.ToString(), Center(on ? UITheme.Label : UITheme.Hint));
            }
            GUI.Label(new Rect(x + 8f * (cell + gap) + 18f, y + 12f, 320f, 24f),
                f.LastAction == f.CurrentPhase
                    ? "steady"
                    : $"transitioning -> {f.LastAction}", UITheme.Hint);
            return y + cell + 26f;
        }

        private static float DrawMovementBars(float x, float y, float w, string title, List<float> values)
        {
            GUI.Label(new Rect(x, y, w, UITheme.LineH(UITheme.Heading)), title, UITheme.Heading);
            y += 34f;
            if (values == null || values.Count == 0)
            {
                GUI.Label(new Rect(x, y, w, 24f), "no data in this frame", UITheme.Hint);
                return y + 30f;
            }

            var max = 1f;
            foreach (var v in values) max = Mathf.Max(max, v);

            const float rowH = 26f, gap = 4f, labelW = 104f;
            for (var i = 0; i < values.Count; i++)
            {
                var r = new Rect(x, y + i * (rowH + gap), w, rowH);
                var name = i < MovementNames.Length ? MovementNames[i] : "M" + i;
                GUI.Label(new Rect(r.x, r.y, labelW, rowH), name, UITheme.Hint);

                var trackX = r.x + labelW;
                var trackW = r.width - labelW - 74f;
                GUI.Box(new Rect(trackX, r.y + 3f, trackW, rowH - 6f), GUIContent.none, UITheme.Panel);

                var prev = GUI.color;
                GUI.color = UITheme.Accent;
                GUI.Box(new Rect(trackX, r.y + 3f, trackW * (values[i] / max), rowH - 6f),
                    GUIContent.none, UITheme.Panel);
                GUI.color = prev;

                GUI.Label(new Rect(trackX + trackW + 12f, r.y, 62f, rowH),
                    values[i].ToString("0"), UITheme.Hint);
            }
            return y + values.Count * (rowH + gap) + 18f;
        }

        private static float DrawForecast(float x, float y, float w, DashboardFrame f)
        {
            GUI.Label(new Rect(x, y, w, UITheme.LineH(UITheme.Heading)), "LSTM forecast", UITheme.Heading);
            y += 34f;
            if (f.ForecastNext30s == null || f.ForecastNext30s.Count == 0)
            {
                GUI.Label(new Rect(x, y, w, 20f),
                    "not available for this controller (no forecaster attached)", UITheme.Hint);
                return y + 30f;
            }

            // 36 values = 3 horizons x 12 movements; the first horizon is the one with any skill.
            var horizon = new List<float>();
            for (var i = 0; i < 12 && i < f.ForecastNext30s.Count; i++) horizon.Add(f.ForecastNext30s[i]);
            GUI.Label(new Rect(x, y, w, 22f),
                "predicted queue 60 s ahead - the pre-registered ablation found this forecast "
                + "significantly DEGRADES the agent; it is shown, not relied on.", UITheme.Hint);
            return DrawMovementBars(x, y + 26f, w, "", horizon);
        }

        /// <summary>How tall the scrolled content is. Kept in step with Draw's panel list.</summary>
        private float ContentHeight(DashboardFrame f)
        {
            var movements = f.QueueLengths != null ? f.QueueLengths.Count : 12;
            var pressures = f.Pressures != null ? f.Pressures.Count : 12;
            var forecast = f.ForecastNext30s != null && f.ForecastNext30s.Count > 0 ? 12 : 0;
            var bars = (movements + pressures + forecast) * 30f + 260f;
            // + a line of slack so the final row is never flush against the bottom edge.
            return 190f + bars + _replayH + 24f;
        }

        /// <summary>
        /// The run archive, and the controls to start a new one.
        ///
        /// Both are here because they are here in the React dashboard, and for the same reason:
        /// this is the screen someone stands in front of during a demo, so "run Webster on this
        /// scene, now run the agent on the same seed" has to be possible without leaving it.
        /// </summary>
        private float DrawReplay(float x, float y, float w)
        {
            GUI.Label(new Rect(x, y, w, UITheme.LineH(UITheme.Heading)), "Session + replay", UITheme.Heading);
            y += 34f;

            if (Session != null && Api != null)
            {
                var scenarios = Api.Scenarios.Count > 0 ? Api.Scenarios : new List<string> { Config.Scenario };
                var controllers = Api.Controllers.Count > 0 ? Api.Controllers : new List<string> { Config.Controller };

                // 34f cut the descenders off "dqn-hybrid", "sel/plain" and "max_pressure".
                // The row pitch has to grow with the button or the rows collide.
                var btnH = UITheme.LineH(UITheme.Button);
                var rowGap = 8f;

                GUI.Label(new Rect(x, y + 4f, 110f, 28f), "scene", UITheme.Hint);
                var cell = Mathf.Min(104f, (w - 130f) / Mathf.Max(1, scenarios.Count));
                for (var i = 0; i < scenarios.Count; i++)
                {
                    if (GUI.Button(new Rect(x + 120f + i * (cell + 4f), y, cell, btnH), scenarios[i],
                            scenarios[i] == Config.Scenario ? UITheme.ButtonOn : UITheme.Button))
                    {
                        Config.Scenario = scenarios[i];
                    }
                }
                y += btnH + rowGap;

                GUI.Label(new Rect(x, y + 4f, 110f, 28f), "controller", UITheme.Hint);
                var ccell = Mathf.Min(170f, (w - 130f) / Mathf.Max(1, controllers.Count));
                for (var i = 0; i < controllers.Count; i++)
                {
                    if (GUI.Button(new Rect(x + 120f + i * (ccell + 4f), y, ccell, btnH), controllers[i],
                            controllers[i] == Config.Controller ? UITheme.ButtonOn : UITheme.Button))
                    {
                        Config.Controller = controllers[i];
                    }
                }
                y += btnH + rowGap + 2f;

                var live = Session.State == "running" || Session.State == "starting";
                if (GUI.Button(new Rect(x, y, 200f, btnH), live ? "Restart run" : "Run this",
                        Session.Busy ? UITheme.ButtonOff : UITheme.Button) && !Session.Busy)
                {
                    Session.Switch(Config);
                }
                if (GUI.Button(new Rect(x + 214f, y, 150f, btnH), "Stop",
                        live ? UITheme.Button : UITheme.ButtonOff) && live)
                {
                    Session.Stop();
                }
                GUI.Label(new Rect(x + 380f, y + 8f, w - 380f, 26f),
                    "hub says: " + Session.State + "   " + Session.RunningController + " / "
                    + Session.RunningScenario, UITheme.Hint);
                y += btnH + rowGap;

                if (!string.IsNullOrEmpty(Session.LastError))
                {
                    GUI.Label(new Rect(x, y, w, 26f), Session.LastError, UITheme.Bad);
                    y += 30f;
                }
            }

            if (GUI.Button(new Rect(x, y, 260f, UITheme.LineH(UITheme.Button)),
                    _showReplay ? "Hide recorded runs" : "Show recorded runs", UITheme.Button))
            {
                _showReplay = !_showReplay;
                if (_showReplay) Api?.RefreshRuns();
            }
            y += 46f;
            if (!_showReplay || Api == null) return y;

            var listH = 150f;
            var rowH = UITheme.LineH(UITheme.Button);   // 30f clipped these the same way
            var pitch = rowH + 4f;
            var view = new Rect(x, y, w * 0.52f, listH);
            var content = new Rect(0f, 0f, view.width - 20f, Api.Runs.Count * pitch + 4f);
            _runScroll = GUI.BeginScrollView(view, _runScroll, content);
            for (var i = 0; i < Api.Runs.Count; i++)
            {
                var run = Api.Runs[i];
                var on = run.RunId == Api.SelectedRunId;
                if (GUI.Button(new Rect(0f, i * pitch, content.width, rowH),
                        run.Name + "   " + run.Mode + "   " + run.CreatedAt,
                        on ? UITheme.ButtonOn : UITheme.Button))
                {
                    Api.SelectRun(run.RunId);
                }
            }
            GUI.EndScrollView();

            // KPI rows for whichever run is selected.
            var kx = x + w * 0.54f;
            if (Api.SelectedRunId == null)
            {
                GUI.Label(new Rect(kx, y, w * 0.46f, 26f), "select a run to see its KPIs", UITheme.Hint);
            }
            else if (Api.RunKpis.Count == 0)
            {
                GUI.Label(new Rect(kx, y, w * 0.46f, 52f),
                    "no KPI rows for this run - live and incomplete episodes have none", UITheme.Wrap);
            }
            else
            {
                GUI.Label(new Rect(kx, y, w * 0.46f, 24f),
                    "scenario   seed   wait      thr.", UITheme.Hint);
                for (var i = 0; i < Api.RunKpis.Count && i < 4; i++)
                {
                    var k = Api.RunKpis[i];
                    GUI.Label(new Rect(kx, y + 26f + i * 26f, w * 0.46f, 24f),
                        k.Scenario + "   " + k.Seed + "   "
                        + (k.AvgWait.HasValue ? k.AvgWait.Value.ToString("0.0") : "-") + " s   "
                        + (k.Throughput.HasValue ? k.Throughput.Value.ToString("0") : "-")
                        + (k.Gridlocked ? "   GRIDLOCK" : ""),
                        UITheme.CellBlurb(false));
                }
            }
            return y + listH + 10f;
        }

        private static GUIStyle Center(GUIStyle from)
        {
            return new GUIStyle(from) { alignment = TextAnchor.MiddleCenter };
        }
    }
}
