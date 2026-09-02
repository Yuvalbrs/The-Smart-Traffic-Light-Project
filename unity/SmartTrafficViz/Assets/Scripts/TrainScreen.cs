// The Train screen: start a training run, watch it learn, then evaluate what came out.
//
// The same three steps the React Train tab offers, against the same endpoints - POST /training,
// GET /training/current, GET /models, POST /evaluation. Nothing is computed here; a second copy
// of "how far along is this run" on the client is a second thing that can disagree with the hub.
//
// The reward curve is drawn by hand because IMGUI has no chart widget. It is a polyline in a box,
// which is all a learning curve needs to be: what the audience has to see is that the line moves,
// and in which direction.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public class TrainScreen
    {
        private static readonly string[] Variants = { "plain", "hybrid", "random-lstm" };
        private static readonly string[] VariantLabels =
        {
            "plain DQN", "DQN + forecast (hybrid)", "DQN + random LSTM (ablation)",
        };

        private readonly HubApi _api;

        private int _variant;
        private int _seed = 42;
        private int _episodes = 30;
        private bool _measured;             // real Hangzhou demand instead of the synthetic rotation
        private string _label = "";
        private Vector2 _modelScroll;
        private string _evalFor;            // model id the evaluate row is open for
        private int _evalScenario;
        private int _evalSeeds = 5;
        private float _nextPoll;

        public TrainScreen(HubApi api)
        {
            _api = api;
            _api.RefreshControllers();
            _api.RefreshModels();
            _api.RefreshTraining();
            _api.RefreshEvaluation();
        }

        /// <summary>Poll the two jobs. Called from Update, not OnGUI, which runs several times a frame.</summary>
        public void Tick()
        {
            if (Time.unscaledTime < _nextPoll) return;
            _nextPoll = Time.unscaledTime + 1f;
            _api.RefreshTraining();
            _api.RefreshEvaluation();

            // A finished run produces a checkpoint that only appears in /models once it lands.
            if (_api.Training != null && _api.Training.Status == "done") _api.RefreshModels();
        }

        public void Draw(Rect box)
        {
            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var x = box.x + 26f;
            var w = box.width - 52f;
            var y = box.y + 22f;

            GUI.Label(new Rect(x, y, w, 52f), "Train a controller", UITheme.Title);
            y += 64f;

            var running = _api.Training != null && _api.Training.Running;

            // --- the form -------------------------------------------------------------
            y = Row(x, y, w, "variant",
                () => _variant = Picker(new Rect(x + 220f, y, w - 220f, 40f), VariantLabels, _variant, running));
            y = Row(x, y, w, "seed",
                () => _seed = IntField(new Rect(x + 220f, y, 160f, 40f), _seed, running));
            y = Row(x, y, w, "episodes",
                () => _episodes = IntField(new Rect(x + 220f, y, 160f, 40f), _episodes, running));
            y = Row(x, y, w, "training demand",
                () => _measured = Picker(new Rect(x + 220f, y, w - 220f, 40f),
                    new[] { "synthetic scenarios (SCN-01/02/03)", "real measured - Hangzhou (SCN-R1)" },
                    _measured ? 1 : 0, running) == 1);
            y = Row(x, y, w, "label (optional)",
                () => _label = GUI.TextField(new Rect(x + 220f, y, w - 220f, 40f), _label ?? "", 40, UITheme.Field));

            GUI.Label(new Rect(x, y, w, 26f),
                _measured
                    ? "The arrival pattern is real measured traffic; the agent learns by controlling it. "
                      + "A demonstration - SCN-R1 sits outside the pre-registered campaign."
                    : "30 episodes is a quick demo run; 300 is the full protocol (~13 min).",
                UITheme.Hint);
            y += 34f;

            if (GUI.Button(new Rect(x, y, 220f, 46f), "Start training",
                    running || _api.Busy ? UITheme.ButtonOff : UITheme.Button) && !running && !_api.Busy)
            {
                _api.StartTraining(Variants[_variant], _seed, _episodes, _measured, _label);
            }
            if (GUI.Button(new Rect(x + 236f, y, 160f, 46f), "Stop",
                    running ? UITheme.Button : UITheme.ButtonOff) && running)
            {
                _api.StopTraining();
            }
            y += 58f;

            if (!string.IsNullOrEmpty(_api.LastError))
            {
                GUI.Label(new Rect(x, y, w, 26f), _api.LastError, UITheme.Bad);
                y += 30f;
            }

            // --- progress -------------------------------------------------------------
            var job = _api.Training;
            if (job == null)
            {
                GUI.Label(new Rect(x, y, w, 26f), "no training job started yet", UITheme.Hint);
                y += 34f;
            }
            else
            {
                GUI.Label(new Rect(x, y, w, 26f),
                    job.Detail + "   -   " + job.Status + "   -   " + job.Done + "/" + job.Total
                    + " episodes"
                    + (job.TrainScenarios != null ? "   -   measured demand" : ""),
                    UITheme.Heading);
                y += 32f;
                Bar(new Rect(x, y, w, 24f), (float)(job.Pct / 100d));
                y += 34f;

                if (job.Curve != null && job.Curve.Count > 1)
                {
                    Curve(new Rect(x, y, w, 150f), job.Curve);
                    y += 160f;
                }
                if (!string.IsNullOrEmpty(job.Error))
                {
                    GUI.Label(new Rect(x, y, w, 26f), job.Error, UITheme.Bad);
                    y += 30f;
                }
            }

            // --- models + evaluation --------------------------------------------------
            GUI.Label(new Rect(x, y, w, 28f), "Trained models", UITheme.Heading);
            y += 34f;

            var listH = Mathf.Max(120f, box.yMax - y - 30f);
            var view = new Rect(x, y, w, listH);
            var rowH = 44f;
            var content = new Rect(0f, 0f, w - 24f, _api.Models.Count * rowH + 10f);
            _modelScroll = GUI.BeginScrollView(view, _modelScroll, content);
            for (var i = 0; i < _api.Models.Count; i++)
            {
                var m = _api.Models[i];
                var r = new Rect(0f, i * rowH, content.width, rowH - 4f);
                GUI.Label(new Rect(r.x + 6f, r.y, r.width * 0.42f, r.height),
                    m.Label + (m.IsUser ? "   [yours]" : ""), UITheme.CellName(m.IsUser));
                GUI.Label(new Rect(r.x + r.width * 0.44f, r.y, 160f, r.height),
                    m.EpisodesTrained + "/" + m.Episodes, UITheme.CellBlurb(false));
                GUI.Label(new Rect(r.x + r.width * 0.58f, r.y, 160f, r.height),
                    m.Source, UITheme.CellBlurb(false));

                if (GUI.Button(new Rect(r.xMax - 170f, r.y + 2f, 160f, r.height - 6f),
                        _evalFor == m.Id ? "Close" : "Evaluate", UITheme.Button))
                {
                    _evalFor = _evalFor == m.Id ? null : m.Id;
                }
            }
            GUI.EndScrollView();
            y += listH + 6f;

            if (_evalFor != null) DrawEvaluate(x, box.yMax - 108f, w);
        }

        private void DrawEvaluate(float x, float y, float w)
        {
            var scenarios = _api.Scenarios.Count > 0 ? _api.Scenarios.ToArray() : new[] { "SCN-05" };
            _evalScenario = Mathf.Clamp(_evalScenario, 0, scenarios.Length - 1);

            GUI.Label(new Rect(x, y, 200f, 34f), "evaluate on", UITheme.Label);
            _evalScenario = Picker(new Rect(x + 200f, y, 420f, 38f), scenarios, _evalScenario, false);
            GUI.Label(new Rect(x + 640f, y, 90f, 34f), "seeds", UITheme.Label);
            _evalSeeds = IntField(new Rect(x + 720f, y, 100f, 38f), _evalSeeds, false);

            var ev = _api.Evaluation;
            var busy = ev != null && ev.Running;
            if (GUI.Button(new Rect(x + 840f, y, 200f, 38f), busy ? "evaluating..." : "Run evaluation",
                    busy ? UITheme.ButtonOff : UITheme.Button) && !busy)
            {
                _api.StartEvaluation(_evalFor, scenarios[_evalScenario], _evalSeeds);
            }

            if (ev != null)
            {
                GUI.Label(new Rect(x, y + 44f, w, 26f),
                    ev.Status == "done"
                        ? "Evaluated " + ev.Detail + " - open Compare to see it beside the campaign."
                        : ev.Detail + " - " + ev.Status + " " + ev.Done + "/" + ev.Total,
                    UITheme.Hint);
            }
        }

        // ------------------------------------------------------------------ widgets

        private static float Row(float x, float y, float w, string label, System.Action draw)
        {
            GUI.Label(new Rect(x, y + 6f, 210f, 32f), label, UITheme.Label);
            draw();
            return y + 50f;
        }

        private static int Picker(Rect r, string[] options, int index, bool disabled)
        {
            var cell = r.width / options.Length;
            for (var i = 0; i < options.Length; i++)
            {
                var on = i == index;
                if (GUI.Button(new Rect(r.x + i * cell, r.y, cell - 6f, r.height), options[i],
                        on ? UITheme.ButtonOn : (disabled ? UITheme.ButtonOff : UITheme.Button))
                    && !disabled)
                {
                    index = i;
                }
            }
            return index;
        }

        private static int IntField(Rect r, int value, bool disabled)
        {
            var text = GUI.TextField(r, value.ToString(), 6, UITheme.Field);
            if (disabled) return value;
            return int.TryParse(text, out var parsed) ? parsed : value;
        }

        private static void Bar(Rect r, float fraction)
        {
            GUI.Box(r, GUIContent.none, UITheme.Panel);
            var prev = GUI.color;
            GUI.color = UITheme.Accent;
            GUI.Box(new Rect(r.x, r.y, r.width * Mathf.Clamp01(fraction), r.height),
                GUIContent.none, UITheme.Panel);
            GUI.color = prev;
        }

        private static void Curve(Rect r, List<float> values)
        {
            GUI.Box(r, GUIContent.none, UITheme.Panel);
            if (values.Count < 2) return;

            float min = values[0], max = values[0];
            foreach (var v in values) { min = Mathf.Min(min, v); max = Mathf.Max(max, v); }
            if (Mathf.Approximately(min, max)) { max = min + 1f; }

            // A polyline of thin boxes: IMGUI has no line primitive, and a chart library for one
            // curve would be a dependency for a screen that shows one number moving.
            var prev = GUI.color;
            GUI.color = UITheme.Accent;
            for (var i = 1; i < values.Count; i++)
            {
                var x0 = r.x + (i - 1) / (float)(values.Count - 1) * r.width;
                var x1 = r.x + i / (float)(values.Count - 1) * r.width;
                var y0 = r.yMax - (values[i - 1] - min) / (max - min) * r.height;
                var y1 = r.yMax - (values[i] - min) / (max - min) * r.height;
                Segment(x0, y0, x1, y1);
            }
            GUI.color = prev;

            GUI.Label(new Rect(r.x + 8f, r.y + 4f, 260f, 24f), "reward  max " + max.ToString("0.0"), UITheme.Hint);
            GUI.Label(new Rect(r.x + 8f, r.yMax - 26f, 260f, 24f), "min " + min.ToString("0.0"), UITheme.Hint);
        }

        private static void Segment(float x0, float y0, float x1, float y1)
        {
            var steps = Mathf.Max(2, Mathf.CeilToInt(Mathf.Abs(x1 - x0)));
            for (var s = 0; s <= steps; s++)
            {
                var t = s / (float)steps;
                GUI.DrawTexture(new Rect(Mathf.Lerp(x0, x1, t), Mathf.Lerp(y0, y1, t), 2f, 2f),
                    Texture2D.whiteTexture);
            }
        }
    }
}
