// What happens when the simulation ends.
//
// Before this existed the only signal that an episode had finished was one line of small text in
// the corner HUD - "episode finished (no more frames)" - over a scene frozen mid-traffic. To
// anyone who was not already looking at that corner, a completed run and a crashed one look
// exactly the same: cars stopped, nothing moving. This turns the end of a run into an event:
// a panel that says it finished, what the run measured, and how that compares.
//
// Three things it deliberately does NOT do:
//
//   * It does not compute a KPI. Every number here comes from the hub's own
//     /runs/{id}/kpis, which is the same extractor the campaign is judged with. A second
//     implementation on this side is how the picture and the numbers start to disagree.
//   * It does not present one episode as if it were a result. The comparison shows this run as a
//     SINGLE SAMPLE against campaign means over 15 seeds, and says so on screen. One episode
//     beating a 15-seed average is luck until it is repeated.
//   * It does not offer a comparison that cannot be made. The confirmatory campaign covers
//     SCN-01..05 only; the other six scenarios have no campaign to compare against, and SCN-R1
//     never will, because it is deliberately outside the confirmatory set. The panel checks
//     before it offers, and explains why when the answer is no.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public class EpisodeSummary
    {
        /// <summary>Scenarios the confirmatory campaign actually covers. Everything else can be
        /// RUN, but there is nothing to compare it against - see scripts/eval_runner.py,
        /// CONFIRMATORY_SCENARIOS. SCN-R1 (real measured demand) is excluded on purpose: running
        /// on it is a demonstration, never evidence, and a test asserts that.</summary>
        private static readonly HashSet<string> Confirmatory =
            new HashSet<string> { "SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05" };

        /// <summary>The episode length every run started from the application uses
        /// (SessionControl's config). The gridlock guard's 10% backlog threshold is calibrated
        /// against it.</summary>
        private const int StandardEpisodeS = 3600;

        private readonly HubApi _api;

        public bool Visible { get; private set; }
        private bool _showCompare;
        private Vector2 _scroll;

        // Captured at the moment the run ended. Reading these off the live session later is a bug
        // waiting to happen: starting the next episode overwrites them.
        private string _runId = "", _scenario = "-", _controller = "-", _endState = "finished";
        private int _seed;
        private double _simTime;

        // "finished" is the SIMULATION's state, not the database's. The hub ends the episode, then
        // extracts the KPIs from the trip-info file and writes the rows - so asking for them the
        // instant the state flips can genuinely arrive before the run exists. Retrying is the
        // difference between a panel that reports a scary "no run with run_id ..." and one that
        // waits the second it needs to.
        private const int MaxRetries = 8;
        private float _retryAt;
        private int _retries;

        public EpisodeSummary(HubApi api) { _api = api; }

        /// <summary>Called once, on the transition into a finished/stopped/failed state.</summary>
        public void Show(string runId, string scenario, string controller, int seed,
                         double simTime, string endState)
        {
            _runId = runId ?? "";
            _scenario = string.IsNullOrEmpty(scenario) ? "-" : scenario;
            _controller = string.IsNullOrEmpty(controller) ? "-" : controller;
            _seed = seed;
            _simTime = simTime;
            _endState = endState;
            _showCompare = false;
            Visible = true;
            _retries = 0;
            _retryAt = Time.unscaledTime + 1.2f;

            if (_runId.Length > 0) _api.SelectRun(_runId);

            // Fetched NOW, not when the button is pressed, so the panel already knows whether a
            // comparison exists by the time the reader looks at it. A button that has to fail
            // before it can tell you it is unavailable is not a button, it is a trap.
            if (Confirmatory.Contains(_scenario)) _api.RefreshComparison(_scenario);
        }

        public void Hide() { Visible = false; }

        /// <summary>Called every frame by the shell; re-asks for the measurements while the hub
        /// is still writing them.</summary>
        public void Tick()
        {
            if (!Visible || _runId.Length == 0) return;
            if (_api.RunKpisLoading || _retries >= MaxRetries) return;
            var haveRows = _api.RunKpis != null && _api.RunKpis.Count > 0;
            if (haveRows) return;
            if (Time.unscaledTime < _retryAt) return;

            _retries++;
            _retryAt = Time.unscaledTime + 1.2f;
            _api.SelectRun(_runId);
        }

        public void Draw(Rect screen)
        {
            if (!Visible) return;

            var w = Mathf.Min(760f, screen.width - 80f);
            var h = Mathf.Min(_showCompare ? 620f : 470f, screen.height - 80f);
            var r = new Rect((screen.width - w) * 0.5f, (screen.height - h) * 0.5f, w, h);

            UITheme.Backdrop(screen);
            GUI.Box(r, GUIContent.none, UITheme.Panel);
            GUILayout.BeginArea(new Rect(r.x + 22f, r.y + 18f, r.width - 44f, r.height - 36f));

            DrawHeader();
            GUILayout.Space(10f);
            _scroll = GUILayout.BeginScrollView(_scroll);
            DrawKpis();
            if (_showCompare) { GUILayout.Space(14f); DrawComparison(); }
            GUILayout.EndScrollView();
            GUILayout.Space(10f);
            DrawButtons();

            GUILayout.EndArea();
        }

        private void DrawHeader()
        {
            // "Stopped" and "finished" are different events and must not be reported as one: a run
            // the user cut short has no KPIs, and calling that "complete" would be a lie the panel
            // then has to explain away with an empty table.
            var title = _endState == "finished" ? "Simulation complete"
                      : _endState == "stopped" ? "Simulation stopped"
                      : "Simulation ended: " + _endState;
            GUILayout.Label(title, UITheme.Title);
            GUILayout.Label(
                _controller + "   ·   " + _scenario + "   ·   seed " + _seed +
                "   ·   " + Mathf.RoundToInt((float)_simTime) + " s simulated",
                UITheme.Hint);
        }

        private void DrawKpis()
        {
            GUILayout.Label("This run", UITheme.Heading);

            if (_api.RunKpisLoading)
            {
                GUILayout.Label("Reading the measurements…", UITheme.Hint);
                return;
            }
            if (!string.IsNullOrEmpty(_api.RunKpisError))
            {
                if (_retries < MaxRetries)
                {
                    GUILayout.Label("Waiting for the hub to finish recording this run…", UITheme.Hint);
                    return;
                }
                // Out of retries. The raw message is kept, but prefixed with the thing that is
                // actually true and actionable - the episode ran, the recording did not land.
                GUILayout.Label(
                    "The episode finished, but the hub never recorded it. That usually means the " +
                    "results database was not writable. Details: " + _api.RunKpisError,
                    UITheme.Bad);
                return;
            }
            if (_api.RunKpis == null || _api.RunKpis.Count == 0)
            {
                // The honest explanation, not an empty grid. KPIs come from the trip-info file
                // SUMO writes when an episode ENDS; a run stopped part way through never produces
                // one, so there is genuinely nothing to show and the reason is worth stating.
                GUILayout.Label(
                    _endState == "finished"
                        ? "No measurements were recorded for this run."
                        : "No measurements: KPIs are computed from a completed episode, and this one was stopped early.",
                    UITheme.Hint);
                return;
            }

            var k = _api.RunKpis[0];
            if (k.Gridlocked)
            {
                // The gridlock guard censors an episode whose insertion backlog -
                // (loaded - departed) / loaded - exceeds 10%. That threshold is calibrated for a
                // FULL 3600 s episode, and SUMO loads vehicles ahead of their departure time: end
                // an episode early and everything scheduled for the remaining time is counted as
                // "never departed". Measured across this database, the same statistic averages
                // 0.05 at 3600 s and 0.44 at 120 s - so on a truncated run the flag says the
                // episode was cut short, not that the network jammed.
                //
                // The stored value is NOT touched. The censoring rule is pre-registered and drives
                // the confirmatory analysis; re-deriving it after seeing a result is precisely the
                // move this project exists to avoid. What is fixed here is the READING of it.
                if (_simTime > 0d && _simTime < StandardEpisodeS * 0.95d)
                {
                    GUILayout.Label(
                        "Flagged gridlock-censored, but this run covered only " +
                        Mathf.RoundToInt((float)_simTime) + " s of the standard " + StandardEpisodeS +
                        " s. The guard is calibrated for full-length episodes, so on a shortened " +
                        "run this flag is unreliable - it reflects the early cut-off, not a jam.",
                        UITheme.Wrap);
                }
                else
                {
                    GUILayout.Label(
                        "GRIDLOCK-CENSORED — the network jammed, so these numbers describe a broken run.",
                        UITheme.Bad);
                }
                GUILayout.Space(4f);
            }

            Metric("Average wait", k.AvgWait, "s / vehicle", true);
            Metric("Average queue", k.AvgQueue, "vehicles", true);
            Metric("Throughput", k.Throughput, "veh / h", false);
            Metric("Stops per vehicle", k.NumStops, "", true);
            Metric("95th-percentile wait", k.WaitP95, "s", true);
            Metric("Fairness (spread)", k.FairnessStd, "s", true);
            Metric("Worst movement", k.WorstMovementMaxWait, "s", true);
        }

        private static void Metric(string label, double? value, string unit, bool lowerIsBetter)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, UITheme.Label, GUILayout.Width(210f));
            GUILayout.Label(value.HasValue ? value.Value.ToString("0.00") : "—",
                            UITheme.Field, GUILayout.Width(90f));
            GUILayout.Label(unit + (unit.Length > 0 ? "   " : "") +
                            (lowerIsBetter ? "lower is better" : "higher is better"), UITheme.Hint);
            GUILayout.EndHorizontal();
        }

        private void DrawComparison()
        {
            GUILayout.Label("Against the campaign on " + _scenario, UITheme.Heading);

            if (!Confirmatory.Contains(_scenario))
            {
                // Say WHY, not just "no data". The reason is a design decision, and an examiner
                // asking "why can't it compare here?" should get the real answer off the screen.
                GUILayout.Label(
                    _scenario == "SCN-R1"
                        ? "SCN-R1 uses real measured demand and sits deliberately OUTSIDE the confirmatory campaign — " +
                          "running on it is a demonstration, never evidence. There is no campaign to compare against, by design."
                        : "The confirmatory campaign covers SCN-01 to SCN-05 only. " + _scenario +
                          " can be run and measured, but no campaign was recorded on it, so there is nothing to compare against.",
                    UITheme.Wrap);
                return;
            }

            if (_api.CompareLoading) { GUILayout.Label("Loading the campaign…", UITheme.Hint); return; }
            if (!string.IsNullOrEmpty(_api.CompareError))
            {
                GUILayout.Label(_api.CompareError, UITheme.Bad);
                return;
            }
            if (_api.CompareRows == null || _api.CompareRows.Count == 0)
            {
                GUILayout.Label("The campaign has no episodes recorded on " + _scenario + ".", UITheme.Hint);
                return;
            }

            var mine = (_api.RunKpis != null && _api.RunKpis.Count > 0) ? _api.RunKpis[0] : null;

            GUILayout.BeginHorizontal();
            GUILayout.Label("Controller", UITheme.Label, GUILayout.Width(230f));
            GUILayout.Label("Avg wait", UITheme.Label, GUILayout.Width(100f));
            GUILayout.Label("Episodes", UITheme.Hint);
            GUILayout.EndHorizontal();

            foreach (var row in _api.CompareRows)
            {
                double? v = null;
                if (row.Values.ContainsKey("avg_waiting_time")) v = row.Values["avg_waiting_time"];
                GUILayout.BeginHorizontal();
                GUILayout.Label(row.Label + (row.IsUserModel ? "  (in-app)" : ""),
                                row.IsOurs ? UITheme.Field : UITheme.Label, GUILayout.Width(230f));
                GUILayout.Label(v.HasValue ? v.Value.ToString("0.00") : "—",
                                UITheme.Label, GUILayout.Width(100f));
                GUILayout.Label(row.Episodes + " episodes", UITheme.Hint);
                GUILayout.EndHorizontal();
            }

            GUILayout.Space(6f);
            if (mine != null && mine.AvgWait.HasValue)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("→ THIS RUN (" + _controller + ")", UITheme.Good, GUILayout.Width(230f));
                GUILayout.Label(mine.AvgWait.Value.ToString("0.00"), UITheme.Good, GUILayout.Width(100f));
                GUILayout.Label("1 episode", UITheme.Hint);
                GUILayout.EndHorizontal();
            }

            GUILayout.Space(8f);
            // The single most important sentence on this panel. Without it the layout invites
            // exactly the wrong reading - that one good episode beat the campaign.
            GUILayout.Label(
                "This run is ONE episode on one seed. The campaign rows are means over 15 held-out seeds. " +
                "A single episode above or below a campaign mean is variance, not a result.",
                UITheme.Wrap);

            if (!string.IsNullOrEmpty(_api.CompareNote))
            {
                GUILayout.Space(4f);
                GUILayout.Label(_api.CompareNote, UITheme.Hint);
            }
        }

        private void DrawButtons()
        {
            GUILayout.BeginHorizontal();

            var canCompare = Confirmatory.Contains(_scenario);
            var compareLabel = _showCompare ? "Hide comparison"
                             : canCompare ? "Compare on this scene"
                             : "Why can't I compare?";
            if (GUILayout.Button(compareLabel, _showCompare ? UITheme.ButtonOn : UITheme.Button,
                                 GUILayout.Height(34f)))
            {
                _showCompare = !_showCompare;
                if (_showCompare && canCompare) _api.RefreshComparison(_scenario);
            }

            if (GUILayout.Button("Close", UITheme.Button, GUILayout.Height(34f))) Hide();

            GUILayout.EndHorizontal();
            GUILayout.Label("The scene stays on its last frame behind this panel.", UITheme.Hint);
        }
    }
}
