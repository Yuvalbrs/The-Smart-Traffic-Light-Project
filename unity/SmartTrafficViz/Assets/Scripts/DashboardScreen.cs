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
            const float pad = 24f;
            var w = Mathf.Min(980f, UnityEngine.Screen.width - pad * 2f);
            var box = new Rect((UnityEngine.Screen.width - w) / 2f, pad, w,
                UnityEngine.Screen.height - pad * 2f);
            GUI.Box(box, GUIContent.none, UITheme.Panel);

            var x = box.x + 20f;
            var y = box.y + 18f;
            var inner = w - 40f;

            GUI.Label(new Rect(x, y, inner, 30f), "Live dashboard", UITheme.Title);
            y += 36f;

            if (_latest == null)
            {
                GUI.Label(new Rect(x, y, inner, 22f),
                    _feed.Connected
                        ? "Connected. Waiting for a frame - start an episode from Controls."
                        : "Not connected to the hub. Is it running on " + _feed.Url + " ?",
                    UITheme.Hint);
                if (!string.IsNullOrEmpty(_feed.LastError))
                {
                    GUI.Label(new Rect(x, y + 24f, inner, 22f), _feed.LastError, UITheme.Hint);
                }
                return;
            }

            y = DrawKpiTiles(x, y, inner, _latest);
            y = DrawPhase(x, y, inner, _latest);
            y = DrawMovementBars(x, y, inner, "Queue per movement (vehicles)", _latest.QueueLengths);
            y = DrawForecast(x, y, inner, _latest);

            GUI.Label(new Rect(x, box.yMax - 30f, inner, 20f),
                $"sim t={_latest.SimTime:0}s   frame #{_latest.Seq}   received={_feed.Received}   " +
                $"dropped={_feed.Dropped}", UITheme.Hint);
        }

        private static float DrawKpiTiles(float x, float y, float w, DashboardFrame f)
        {
            const float h = 62f;
            var third = (w - 16f) / 3f;
            DrawTile(new Rect(x, y, third, h), "avg wait so far",
                $"{f.RunningKpis.AvgWaitSoFar:0.0} s");
            DrawTile(new Rect(x + third + 8f, y, third, h), "throughput so far",
                $"{f.RunningKpis.ThroughputSoFar:0} veh/h");
            DrawTile(new Rect(x + (third + 8f) * 2f, y, third, h), "queue now",
                $"{f.RunningKpis.CurrentQueueTotal:0} veh");
            GUI.Label(new Rect(x, y + h + 2f, w, 18f),
                "Running estimates, not the confirmatory KPIs - those come from trip-info after "
                + "the episode ends.", UITheme.Hint);
            return y + h + 26f;
        }

        private static void DrawTile(Rect r, string label, string value)
        {
            GUI.Box(r, GUIContent.none, UITheme.Panel);
            GUI.Label(new Rect(r.x + 10f, r.y + 6f, r.width - 20f, 18f), label, UITheme.Hint);
            GUI.Label(new Rect(r.x + 10f, r.y + 24f, r.width - 20f, 30f), value, UITheme.Heading);
        }

        private static float DrawPhase(float x, float y, float w, DashboardFrame f)
        {
            GUI.Label(new Rect(x, y, w, 22f), "Signal phase", UITheme.Heading);
            y += 24f;
            const float cell = 30f, gap = 6f;
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
            GUI.Label(new Rect(x + 8f * (cell + gap) + 12f, y + 6f, 240f, 20f),
                f.LastAction == f.CurrentPhase
                    ? "steady"
                    : $"transitioning -> {f.LastAction}", UITheme.Hint);
            return y + cell + 18f;
        }

        private static float DrawMovementBars(float x, float y, float w, string title, List<float> values)
        {
            GUI.Label(new Rect(x, y, w, 22f), title, UITheme.Heading);
            y += 24f;
            if (values == null || values.Count == 0)
            {
                GUI.Label(new Rect(x, y, w, 20f), "no data in this frame", UITheme.Hint);
                return y + 24f;
            }

            var max = 1f;
            foreach (var v in values) max = Mathf.Max(max, v);

            const float rowH = 18f, gap = 3f, labelW = 74f;
            for (var i = 0; i < values.Count; i++)
            {
                var r = new Rect(x, y + i * (rowH + gap), w, rowH);
                var name = i < MovementNames.Length ? MovementNames[i] : "M" + i;
                GUI.Label(new Rect(r.x, r.y, labelW, rowH), name, UITheme.Hint);

                var trackX = r.x + labelW;
                var trackW = r.width - labelW - 52f;
                GUI.Box(new Rect(trackX, r.y + 3f, trackW, rowH - 6f), GUIContent.none, UITheme.Panel);

                var prev = GUI.color;
                GUI.color = UITheme.Accent;
                GUI.Box(new Rect(trackX, r.y + 3f, trackW * (values[i] / max), rowH - 6f),
                    GUIContent.none, UITheme.Panel);
                GUI.color = prev;

                GUI.Label(new Rect(trackX + trackW + 8f, r.y, 44f, rowH),
                    values[i].ToString("0"), UITheme.Hint);
            }
            return y + values.Count * (rowH + gap) + 12f;
        }

        private static float DrawForecast(float x, float y, float w, DashboardFrame f)
        {
            GUI.Label(new Rect(x, y, w, 22f), "LSTM forecast", UITheme.Heading);
            y += 24f;
            if (f.ForecastNext30s == null || f.ForecastNext30s.Count == 0)
            {
                GUI.Label(new Rect(x, y, w, 20f),
                    "not available for this controller (no forecaster attached)", UITheme.Hint);
                return y + 24f;
            }

            // 36 values = 3 horizons x 12 movements; the first horizon is the one with any skill.
            var horizon = new List<float>();
            for (var i = 0; i < 12 && i < f.ForecastNext30s.Count; i++) horizon.Add(f.ForecastNext30s[i]);
            GUI.Label(new Rect(x, y, w, 18f),
                "predicted queue 60 s ahead - the pre-registered ablation found this forecast "
                + "significantly DEGRADES the agent; it is shown, not relied on.", UITheme.Hint);
            return DrawMovementBars(x, y + 20f, w, "", horizon);
        }

        private static GUIStyle Center(GUIStyle from)
        {
            return new GUIStyle(from) { alignment = TextAnchor.MiddleCenter };
        }
    }
}
