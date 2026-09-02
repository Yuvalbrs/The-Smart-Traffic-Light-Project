// The Compare screen: every controller on one scenario, side by side.
//
// A read-only view of GET /comparison - the same rows, the same winners, the same caveats the
// React tab shows, because they come from the same endpoint. Nothing is recomputed here.
//
// Two rules carried over deliberately, because both are about not overclaiming:
//
//   - A user-trained model is shown but can never win a column. It is usually evaluated on far
//     fewer seeds and on different code, so a model trained for thirty seconds can top a column
//     against a 900-episode campaign purely by variance - which reads as a result and is not one.
//   - When the request fails the rows are cleared, not left on screen. Showing the previous
//     scenario's table under a "no episodes recorded for this one" message lets one scenario's
//     numbers be read as another's.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public class CompareScreen
    {
        private readonly HubApi _api;
        private int _scenario;
        private string _loadedFor;
        private Vector2 _scroll;

        public CompareScreen(HubApi api)
        {
            _api = api;
            _api.RefreshControllers();
        }

        public void Draw(Rect box)
        {
            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var x = box.x + 26f;
            var w = box.width - 52f;
            var y = box.y + 22f;

            GUI.Label(new Rect(x, y, w, 52f), "Controller comparison", UITheme.Title);
            y += 64f;

            var scenarios = _api.Scenarios.Count > 0
                ? _api.Scenarios
                : new List<string> { "SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05" };
            // Default to the scenario the campaign headline is quoted on.
            if (_loadedFor == null)
            {
                var i = scenarios.IndexOf("SCN-05");
                _scenario = i >= 0 ? i : 0;
            }
            _scenario = Mathf.Clamp(_scenario, 0, scenarios.Count - 1);

            GUI.Label(new Rect(x, y + 6f, 130f, 32f), "scenario", UITheme.Label);
            var cell = Mathf.Min(112f, (w - 140f) / scenarios.Count);
            for (var i = 0; i < scenarios.Count; i++)
            {
                if (GUI.Button(new Rect(x + 140f + i * (cell + 4f), y, cell, 40f), scenarios[i],
                        i == _scenario ? UITheme.ButtonOn : UITheme.Button))
                {
                    _scenario = i;
                    _loadedFor = null;
                }
            }
            y += 52f;

            if (_loadedFor != scenarios[_scenario])
            {
                _loadedFor = scenarios[_scenario];
                _api.RefreshComparison(_loadedFor);
            }

            if (!string.IsNullOrEmpty(_api.CompareError))
            {
                GUI.Label(new Rect(x, y, w, 60f), _api.CompareError, UITheme.Bad);
                return;
            }
            if (_api.CompareRows.Count == 0)
            {
                GUI.Label(new Rect(x, y, w, 30f), "loading...", UITheme.Hint);
                return;
            }

            var kpis = _api.CompareKpis;
            var rows = _api.CompareRows;
            var best = BestValues(rows, kpis);

            // --- header ---------------------------------------------------------------
            var nameW = 300f;
            var colW = Mathf.Max(120f, (w - nameW - 130f) / Mathf.Max(1, kpis.Count));
            GUI.Label(new Rect(x, y, nameW, 26f), "controller", UITheme.Hint);
            GUI.Label(new Rect(x + nameW, y, 120f, 26f), "episodes", UITheme.Hint);
            for (var c = 0; c < kpis.Count; c++)
            {
                GUI.Label(new Rect(x + nameW + 130f + c * colW, y, colW - 8f, 44f),
                    kpis[c].Label + (kpis[c].LowerIsBetter ? "  v" : "  ^"), UITheme.Hint);
            }
            y += 48f;

            // --- rows -----------------------------------------------------------------
            const float rowH = 40f;
            var listH = Mathf.Min(rows.Count * rowH + 8f, box.yMax - y - 190f);
            var view = new Rect(x, y, w, listH);
            var content = new Rect(0f, 0f, w - 20f, rows.Count * rowH + 4f);
            _scroll = GUI.BeginScrollView(view, _scroll, content);
            for (var r = 0; r < rows.Count; r++)
            {
                var row = rows[r];
                var rr = new Rect(0f, r * rowH, content.width, rowH - 3f);
                if (row.IsOurs)
                {
                    var prev = GUI.color;
                    GUI.color = new Color(UITheme.Accent.r, UITheme.Accent.g, UITheme.Accent.b, 0.18f);
                    GUI.Box(rr, GUIContent.none, UITheme.Panel);
                    GUI.color = prev;
                }

                GUI.Label(new Rect(rr.x + 8f, rr.y, nameW, rowH),
                    row.Label + (row.IsUserModel ? "  [yours]" : ""), UITheme.CellName(row.IsOurs));
                GUI.Label(new Rect(rr.x + nameW, rr.y, 120f, rowH),
                    row.Episodes + "  (" + (row.GridlockRate * 100d).ToString("0") + "% gridlock)",
                    UITheme.CellBlurb(false));

                for (var c = 0; c < kpis.Count; c++)
                {
                    var k = kpis[c];
                    row.Values.TryGetValue(k.Key, out var v);
                    var isBest = v.HasValue && !row.IsUserModel
                                 && best.TryGetValue(k.Key, out var b)
                                 && Mathf.Approximately((float)v.Value, (float)b);
                    GUI.Label(new Rect(rr.x + nameW + 130f + c * colW, rr.y, colW - 8f, rowH),
                        v.HasValue ? Format(k.Key, v.Value) : "-",
                        isBest ? UITheme.Good : UITheme.CellBlurb(false));
                }
            }
            GUI.EndScrollView();
            y += listH + 10f;

            // --- the honest note ------------------------------------------------------
            if (!string.IsNullOrEmpty(_api.CompareNote))
            {
                GUI.Label(new Rect(x, y, w, 80f), _api.CompareNote, UITheme.Wrap);
                y += 84f;
            }

            // --- one chart, on the headline KPI ---------------------------------------
            if (kpis.Count > 0 && box.yMax - y > 90f)
            {
                Bars(new Rect(x, y, w, box.yMax - y - 16f), rows, kpis[0]);
            }
        }

        /// <summary>Best value per KPI over the CAMPAIGN rows only - user models cannot win.</summary>
        private static Dictionary<string, double> BestValues(List<CompareRow> rows, List<CompareKpi> kpis)
        {
            var best = new Dictionary<string, double>();
            foreach (var k in kpis)
            {
                var have = false;
                var acc = 0d;
                foreach (var row in rows)
                {
                    if (row.IsUserModel) continue;
                    if (!row.Values.TryGetValue(k.Key, out var v) || !v.HasValue) continue;
                    if (!have) { acc = v.Value; have = true; continue; }
                    acc = k.LowerIsBetter ? System.Math.Min(acc, v.Value) : System.Math.Max(acc, v.Value);
                }
                if (have) best[k.Key] = acc;
            }
            return best;
        }

        private static void Bars(Rect r, List<CompareRow> rows, CompareKpi kpi)
        {
            GUI.Label(new Rect(r.x, r.y, r.width, 26f),
                kpi.Label + (kpi.LowerIsBetter ? " - lower is better" : " - higher is better"),
                UITheme.Heading);
            var plot = new Rect(r.x, r.y + 30f, r.width, r.height - 30f);

            var max = 0d;
            foreach (var row in rows)
            {
                if (row.Values.TryGetValue(kpi.Key, out var v) && v.HasValue) max = System.Math.Max(max, v.Value);
            }
            if (max <= 0d) return;

            var slot = plot.width / rows.Count;
            for (var i = 0; i < rows.Count; i++)
            {
                if (!rows[i].Values.TryGetValue(kpi.Key, out var v) || !v.HasValue) continue;
                // Leave headroom for the value label above the tallest bar.
                var h = (float)(v.Value / max) * (plot.height - 62f);
                var bar = new Rect(plot.x + i * slot + 8f, plot.yMax - 34f - h, slot - 16f, h);
                var prev = GUI.color;
                GUI.color = rows[i].IsOurs ? UITheme.Accent : new Color(0.45f, 0.50f, 0.60f, 0.85f);
                GUI.Box(bar, GUIContent.none, UITheme.Panel);
                GUI.color = prev;

                // The number, above its own bar. A bar chart with no scale and no labels tells a
                // viewer which is biggest and nothing else - and "which is biggest" is the one
                // thing the table already said. The value is the reason to draw it.
                GUI.Label(new Rect(plot.x + i * slot, bar.y - 28f, slot, 26f),
                    Format(kpi.Key, v.Value), Centred(UITheme.CellName(rows[i].IsOurs)));
                GUI.Label(new Rect(plot.x + i * slot, plot.yMax - 30f, slot, 26f),
                    Shorten(rows[i].Label), Centred(UITheme.CellBlurb(rows[i].IsOurs)));
            }
        }

        private static GUIStyle Centred(GUIStyle from)
        {
            return new GUIStyle(from) { alignment = TextAnchor.MiddleCenter };
        }

        private static string Format(string key, double v)
        {
            return key == "throughput" ? v.ToString("0") : v.ToString("0.00");
        }

        private static string Shorten(string label)
        {
            return label.Length <= 16 ? label : label.Substring(0, 15) + "...";
        }
    }
}
