// T-05-03 - the scene / controller picker.
//
// "Scene" here means a traffic scenario - SCN-01..SCN-05, the demand patterns defined in
// config/scenarios/. Names and one-line descriptions are mirrored from those files so the menu
// says the same thing the simulation is actually running.
//
// One panel, drawn both as a full menu screen and as an overlay inside the live view, so the
// controls do not drift apart between the two. Only one episode exists at a time (libsumo holds
// a single global simulation), so choosing a different scenario or controller stops the current
// run and starts the new one.

using UnityEngine;

namespace SmartTraffic
{
    public static class RunSetupUI
    {
        public struct Scene
        {
            public string Id, Name, Blurb;
            public Scene(string id, string name, string blurb) { Id = id; Name = name; Blurb = blurb; }
        }

        /// <summary>Mirrors config/scenarios/scn_0*.yaml.</summary>
        public static readonly Scene[] Scenes =
        {
            new Scene("SCN-01", "Uniform light", "Low, symmetric demand. The easy regime."),
            new Scene("SCN-02", "Uniform heavy", "High, symmetric demand. Saturated but stable."),
            new Scene("SCN-03", "Rush hour peak", "N/S peak decays as E/W builds."),
            new Scene("SCN-04", "Asymmetric", "Heavy N/S, light E/W. Starvation probe."),
            new Scene("SCN-05", "Shifting demand", "Both axes oscillate, 90 deg out of phase."),
        };

        /// <summary>Mirrors CONTROLLERS in src/api/live.py.</summary>
        public static readonly string[] Controllers =
        {
            "sel/plain", "dqn-plain", "dqn-hybrid", "webster", "max_pressure", "actuated",
        };

        private static readonly float[] Speeds = { 1f, 2f, 5f, 10f, 0f };

        public const float PanelWidth = 560f;
        public const float PanelHeight = 476f;

        /// <summary>Draws the picker. Returns true if the user asked to run something.</summary>
        public static bool Draw(Rect box, ref RunConfig cfg, SessionControl session, bool showStop)
        {
            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var x = box.x + 20f;
            var w = box.width - 40f;
            var y = box.y + 16f;
            var run = false;

            GUI.Label(new Rect(x, y, w, 24f), "SCENE  (traffic scenario)", UITheme.Heading);
            y += 26f;

            foreach (var scene in Scenes)
            {
                var on = cfg.Scenario == scene.Id;
                var label = scene.Id + "   " + scene.Name + "      " + scene.Blurb;
                if (GUI.Button(new Rect(x, y, w, 30f), label, on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Scenario = scene.Id;
                }
                y += 33f;
            }

            y += 8f;
            GUI.Label(new Rect(x, y, w, 24f), "CONTROLLER  (who drives the lights)", UITheme.Heading);
            y += 26f;

            const int perRow = 3;
            var bw = (w - 8f * (perRow - 1)) / perRow;
            for (var i = 0; i < Controllers.Length; i++)
            {
                var col = i % perRow;
                var row = i / perRow;
                var on = cfg.Controller == Controllers[i];
                var r = new Rect(x + col * (bw + 8f), y + row * 34f, bw, 30f);
                if (GUI.Button(r, Controllers[i], on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Controller = Controllers[i];
                }
            }
            y += 34f * Mathf.Ceil(Controllers.Length / (float)perRow) + 10f;

            GUI.Label(new Rect(x, y, w, 22f), "SPEED  (simulated seconds per real second)", UITheme.Heading);
            y += 24f;
            var sw = (w - 8f * (Speeds.Length - 1)) / Speeds.Length;
            for (var i = 0; i < Speeds.Length; i++)
            {
                var on = Mathf.Approximately(cfg.Speed, Speeds[i]);
                var label = Speeds[i] <= 0f ? "as fast" : Speeds[i] + "x";
                if (GUI.Button(new Rect(x + i * (sw + 8f), y, sw, 28f), label,
                        on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Speed = Speeds[i];
                }
            }
            y += 36f;

            GUI.Label(new Rect(x, y, w, 20f),
                "seed " + cfg.Seed + "   -   same seed + same scene = identical traffic, " +
                "which is what makes two controllers comparable", UITheme.Hint);
            y += 26f;

            var status = session == null
                ? ""
                : "session: " + session.State + "   " + session.RunningController + " on " +
                  session.RunningScenario + "   sim " + session.SimTime.ToString("F0") + " s";
            GUI.Label(new Rect(x, y, w, 20f), status, UITheme.Label);
            y += 24f;

            if (session != null && session.LastError != null)
            {
                var err = new GUIStyle(UITheme.Hint);
                err.normal.textColor = new Color(1f, 0.45f, 0.4f);
                GUI.Label(new Rect(x, y, w, 20f), session.LastError, err);
            }
            y += 22f;

            var busy = session != null && session.Busy;
            GUI.enabled = !busy;
            var runW = showStop ? w * 0.62f : w;
            if (GUI.Button(new Rect(x, y, runW, 36f),
                    busy ? "working..." : "RUN  " + cfg.Scenario + "  /  " + cfg.Controller,
                    UITheme.ButtonOn))
            {
                run = true;
            }
            if (showStop && GUI.Button(new Rect(x + runW + 8f, y, w - runW - 8f, 36f),
                    "Stop", UITheme.Button))
            {
                session?.Stop();
            }
            GUI.enabled = true;

            return run;
        }
    }
}
