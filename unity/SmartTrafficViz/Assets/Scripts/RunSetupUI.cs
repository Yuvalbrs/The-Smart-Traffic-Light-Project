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

        // Layout constants. The panel height is DERIVED from these rather than guessed, because a
        // guessed height is what pushed the run button off the bottom of the panel before.
        private const float Pad = 24f;      // panel inset
        private const float Gap = 7f;       // between buttons in a group
        private const float Section = 16f;  // between groups
        private const float HeadH = 22f;
        private const float RowH = 32f;
        private const float RunH = 40f;

        public const float PanelWidth = 660f;

        public static float PanelHeight =>
            Pad
            + HeadH + Gap + Scenes.Length * (RowH + Gap) + Section        // scenes
            + HeadH + Gap + 2f * (RowH + Gap) + Section                   // controllers (2 rows)
            + HeadH + Gap + (RowH + Gap) + Section                        // speed
            + 20f + 22f + 18f + Gap                                       // seed, status, error
            + RunH + Pad;

        /// <summary>Draws the picker. Returns true if the user asked to run something.</summary>
        public static bool Draw(Rect box, ref RunConfig cfg, SessionControl session, bool showStop)
        {
            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var x = box.x + Pad;
            var w = box.width - Pad * 2f;
            var y = box.y + Pad;
            var run = false;

            GUI.Label(new Rect(x, y, w, HeadH), "SCENE   traffic scenario", UITheme.Heading);
            y += HeadH + Gap;

            foreach (var scene in Scenes)
            {
                var on = cfg.Scenario == scene.Id;
                var row = new Rect(x, y, w, RowH);
                // Button first as the background, then the text in fixed columns on top. A single
                // centred caption made every row start at a different x, so nothing lined up.
                if (GUI.Button(row, GUIContent.none, on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Scenario = scene.Id;
                }
                GUI.Label(new Rect(row.x + 14f, row.y, 66f, RowH), scene.Id, UITheme.CellId(on));
                GUI.Label(new Rect(row.x + 88f, row.y, 132f, RowH), scene.Name, UITheme.CellName(on));
                GUI.Label(new Rect(row.x + 228f, row.y, w - 242f, RowH), scene.Blurb,
                    UITheme.CellBlurb(on));
                y += RowH + Gap;
            }

            y += Section - Gap;
            GUI.Label(new Rect(x, y, w, HeadH), "CONTROLLER   who drives the lights", UITheme.Heading);
            y += HeadH + Gap;

            const int perRow = 3;
            var bw = (w - Gap * (perRow - 1)) / perRow;
            for (var i = 0; i < Controllers.Length; i++)
            {
                var on = cfg.Controller == Controllers[i];
                var r = new Rect(x + i % perRow * (bw + Gap), y + i / perRow * (RowH + Gap), bw, RowH);
                if (GUI.Button(r, Controllers[i], on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Controller = Controllers[i];
                }
            }
            y += 2f * (RowH + Gap) + Section - Gap;

            GUI.Label(new Rect(x, y, w, HeadH), "SPEED   simulated seconds per real second",
                UITheme.Heading);
            y += HeadH + Gap;

            var sw = (w - Gap * (Speeds.Length - 1)) / Speeds.Length;
            for (var i = 0; i < Speeds.Length; i++)
            {
                var on = Mathf.Approximately(cfg.Speed, Speeds[i]);
                var label = Speeds[i] <= 0f ? "as fast" : Speeds[i] + "x";
                if (GUI.Button(new Rect(x + i * (sw + Gap), y, sw, RowH), label,
                        on ? UITheme.ButtonOn : UITheme.Button))
                {
                    cfg.Speed = Speeds[i];
                }
            }
            y += RowH + Gap + Section - Gap;

            GUI.Label(new Rect(x, y, w, 20f),
                "seed " + cfg.Seed + "   -   same seed and scene means identical traffic",
                UITheme.Hint);
            y += 20f;

            var status = session == null
                ? ""
                : "session: " + session.State + "   " + session.RunningController + " on " +
                  session.RunningScenario + "   sim " + session.SimTime.ToString("F0") + " s";
            GUI.Label(new Rect(x, y, w, 22f), status, UITheme.Label);
            y += 22f;

            if (session != null && session.LastError != null)
            {
                var err = new GUIStyle(UITheme.Hint);
                err.normal.textColor = new Color(1f, 0.45f, 0.4f);
                GUI.Label(new Rect(x, y, w, 18f), session.LastError, err);
            }
            y += 18f + Gap;

            var busy = session != null && session.Busy;
            GUI.enabled = !busy;
            var runW = showStop ? w - 130f - Gap : w;
            if (GUI.Button(new Rect(x, y, runW, RunH),
                    busy ? "working..." : "RUN   " + cfg.Scenario + "   /   " + cfg.Controller,
                    UITheme.ButtonOn))
            {
                run = true;
            }
            if (showStop && GUI.Button(new Rect(x + runW + Gap, y, 130f, RunH), "Stop", UITheme.Button))
            {
                session?.Stop();
            }
            GUI.enabled = true;

            return run;
        }
    }
}
