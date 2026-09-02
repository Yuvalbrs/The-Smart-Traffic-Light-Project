// REST client for the endpoints the Unity screens need beyond starting and stopping an episode.
//
// SessionControl already covers /sessions; this covers the rest of what the React dashboard uses:
// the run archive behind the replay browser, the model catalogue, the training and evaluation
// jobs, and the controller comparison. Same shape as SessionControl deliberately - requests run
// on background tasks and park their results in fields the main thread reads during OnGUI, and
// nothing here touches the scene, because Unity API calls are main-thread only.
//
// Every screen in this client is therefore a view over the same hub the browser talks to. That is
// what stops the two disagreeing: there is no second copy of a KPI, a model list or a comparison
// computed on this side.

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SmartTraffic
{
    public sealed class RunRow
    {
        public string RunId, Name, Controller, Mode, CreatedAt;
    }

    public sealed class EpisodeKpiRow
    {
        public string Scenario;
        public int Seed;
        public bool Gridlocked;
        public double? AvgWait, Throughput, AvgQueue, WaitP95;
    }

    public sealed class ModelRow
    {
        public string Id, Label, Variant, Source;
        public int Seed, Episodes, EpisodesTrained;
        public bool HasFinal;
        public bool IsUser => Source == "user";
    }

    public sealed class CompareKpi
    {
        public string Key, Label;
        public bool LowerIsBetter;
    }

    public sealed class CompareRow
    {
        public string Label;
        public bool IsOurs, IsUserModel;
        public int Episodes;
        public double GridlockRate;
        public readonly Dictionary<string, double?> Values = new Dictionary<string, double?>();
    }

    public sealed class JobStatus
    {
        public string JobId, Status, Label, Error;
        public int Done, Total;
        public double Pct;
        public string Detail = "";          // variant/seed, or model/scenario
        public List<string> TrainScenarios;
        /// <summary>Per-episode reward, straight off the hub. The status carries the whole curve
        /// on every poll, so the client never has to accumulate (and drift from) its own copy.</summary>
        public List<float> Curve = new List<float>();
        public bool Running => Status == "running";
    }

    public class HubApi
    {
        // One HttpClient for the process. A per-request client exhausts sockets under polling.
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

        private readonly string _base;

        public string LastError { get; private set; }

        // --- populated by the Refresh* calls, read on the main thread ---
        public List<string> Controllers = new List<string>();
        public List<string> Scenarios = new List<string>();
        public List<RunRow> Runs = new List<RunRow>();
        public List<EpisodeKpiRow> RunKpis = new List<EpisodeKpiRow>();
        public string SelectedRunId;
        public List<ModelRow> Models = new List<ModelRow>();
        public List<CompareKpi> CompareKpis = new List<CompareKpi>();
        public List<CompareRow> CompareRows = new List<CompareRow>();
        public string CompareNote, CompareError;
        public JobStatus Training, Evaluation;
        public bool Busy { get; private set; }

        public HubApi(string wsUrl)
        {
            var http = wsUrl.Replace("ws://", "http://").Replace("wss://", "https://");
            var cut = http.IndexOf("/ws/", StringComparison.Ordinal);
            _base = cut > 0 ? http.Substring(0, cut) : http.TrimEnd('/');
        }

        // ------------------------------------------------------------------ options

        public async void RefreshControllers()
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/controllers");
                if (!resp.IsSuccessStatusCode) return;
                var o = JObject.Parse(await resp.Content.ReadAsStringAsync());
                Controllers = ToStrings(o["controllers"]);
                Scenarios = ToStrings(o["scenarios"]);
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        // ------------------------------------------------------------------ replay

        public async void RefreshRuns(int limit = 40)
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/runs?limit=" + limit + "&offset=0");
                if (!resp.IsSuccessStatusCode) return;
                var o = JObject.Parse(await resp.Content.ReadAsStringAsync());
                var rows = new List<RunRow>();
                foreach (var r in o["runs"] ?? new JArray())
                {
                    rows.Add(new RunRow
                    {
                        RunId = (string)r["run_id"],
                        Name = (string)r["name"],
                        Controller = (string)r["controller"],
                        Mode = (string)r["mode"],
                        CreatedAt = (string)r["created_at"],
                    });
                }
                Runs = rows;
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        public async void SelectRun(string runId)
        {
            SelectedRunId = runId;
            RunKpis = new List<EpisodeKpiRow>();
            try
            {
                var resp = await Http.GetAsync(_base + "/runs/" + Uri.EscapeDataString(runId) + "/kpis");
                if (!resp.IsSuccessStatusCode) return;
                var o = JObject.Parse(await resp.Content.ReadAsStringAsync());
                var rows = new List<EpisodeKpiRow>();
                foreach (var r in o["rows"] ?? new JArray())
                {
                    rows.Add(new EpisodeKpiRow
                    {
                        Scenario = (string)r["scenario"],
                        Seed = (int?)r["seed"] ?? 0,
                        Gridlocked = (bool?)r["gridlock_censored"] ?? false,
                        AvgWait = (double?)r["avg_waiting_time"],
                        Throughput = (double?)r["throughput"],
                        AvgQueue = (double?)r["avg_queue_length"],
                        WaitP95 = (double?)r["wait_p95"],
                    });
                }
                RunKpis = rows;
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        // ------------------------------------------------------------------ models

        public async void RefreshModels()
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/models");
                if (!resp.IsSuccessStatusCode) return;
                var o = JObject.Parse(await resp.Content.ReadAsStringAsync());
                var rows = new List<ModelRow>();
                foreach (var m in o["models"] ?? new JArray())
                {
                    rows.Add(new ModelRow
                    {
                        Id = (string)m["id"],
                        Label = (string)m["label"],
                        Variant = (string)m["variant"],
                        Source = (string)m["source"],
                        Seed = (int?)m["seed"] ?? 0,
                        Episodes = (int?)m["episodes"] ?? 0,
                        EpisodesTrained = (int?)m["episodes_trained"] ?? 0,
                        HasFinal = (bool?)m["has_final"] ?? false,
                    });
                }
                Models = rows;
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        // ------------------------------------------------------------------ training

        public async void StartTraining(string variant, int seed, int episodes, bool measuredDemand,
            string label)
        {
            if (Busy) return;
            Busy = true;
            LastError = null;
            try
            {
                var body = JsonConvert.SerializeObject(new
                {
                    variant,
                    seed,
                    episodes,
                    episode_length_s = (int?)null,
                    label = string.IsNullOrWhiteSpace(label) ? null : label,
                    // Null keeps train_dqn's own default rotation; naming SCN-R1 trains against
                    // the measured Hangzhou counts held in the database.
                    train_scenarios = measuredDemand ? new[] { "SCN-R1" } : null,
                });
                var resp = await Http.PostAsync(_base + "/training",
                    new StringContent(body, Encoding.UTF8, "application/json"));
                var text = await resp.Content.ReadAsStringAsync();
                if (!resp.IsSuccessStatusCode) { LastError = (int)resp.StatusCode + ": " + Detail(text); return; }
                Training = ParseTraining(JObject.Parse(text));
            }
            catch (Exception exc) { LastError = exc.Message; }
            finally { Busy = false; }
        }

        public async void RefreshTraining()
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/training/current");
                // 404 is the ordinary "no job yet" answer and must not be shown as a failure.
                if (!resp.IsSuccessStatusCode) return;
                Training = ParseTraining(JObject.Parse(await resp.Content.ReadAsStringAsync()));
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        public async void StopTraining()
        {
            try { await Http.DeleteAsync(_base + "/training/current"); }
            catch (Exception exc) { LastError = exc.Message; }
        }

        private static JobStatus ParseTraining(JObject o)
        {
            return new JobStatus
            {
                JobId = (string)o["job_id"],
                Status = (string)o["status"],
                Label = (string)o["label"],
                Error = (string)o["error"],
                Done = (int?)o["episodes_done"] ?? 0,
                Total = (int?)o["episodes"] ?? 0,
                Pct = (double?)o["pct"] ?? 0d,
                Detail = (string)o["variant"] + " / seed " + (int?)o["seed"],
                TrainScenarios = o["train_scenarios"] is JArray a ? ToStrings(a) : null,
                Curve = Rewards(o["curve"]),
            };
        }

        // ------------------------------------------------------------------ evaluation

        public async void StartEvaluation(string modelId, string scenario, int seeds)
        {
            if (Busy) return;
            Busy = true;
            LastError = null;
            try
            {
                var body = JsonConvert.SerializeObject(new { model_id = modelId, scenario, seeds });
                var resp = await Http.PostAsync(_base + "/evaluation",
                    new StringContent(body, Encoding.UTF8, "application/json"));
                var text = await resp.Content.ReadAsStringAsync();
                if (!resp.IsSuccessStatusCode) { LastError = (int)resp.StatusCode + ": " + Detail(text); return; }
                Evaluation = ParseEvaluation(JObject.Parse(text));
            }
            catch (Exception exc) { LastError = exc.Message; }
            finally { Busy = false; }
        }

        public async void RefreshEvaluation()
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/evaluation/current");
                if (!resp.IsSuccessStatusCode) return;
                Evaluation = ParseEvaluation(JObject.Parse(await resp.Content.ReadAsStringAsync()));
            }
            catch (Exception exc) { LastError = exc.Message; }
        }

        private static JobStatus ParseEvaluation(JObject o)
        {
            return new JobStatus
            {
                JobId = (string)o["job_id"],
                Status = (string)o["status"],
                Error = (string)o["error"],
                Done = (int?)o["episodes_done"] ?? 0,
                Total = (int?)o["episodes"] ?? 0,
                Pct = (double?)o["pct"] ?? 0d,
                Detail = (string)o["model_label"] ?? (string)o["model_id"],
            };
        }

        // ------------------------------------------------------------------ comparison

        public async void RefreshComparison(string scenario)
        {
            CompareError = null;
            try
            {
                var resp = await Http.GetAsync(_base + "/comparison?scenario=" + Uri.EscapeDataString(scenario));
                var text = await resp.Content.ReadAsStringAsync();
                if (!resp.IsSuccessStatusCode)
                {
                    // Clear the rows as well as setting the error. Leaving the previous scenario's
                    // table up under a "no episodes for SCN-08" message let one scenario's numbers
                    // be read as another's - the same defect this cost us in the React view.
                    CompareError = Detail(text);
                    CompareRows = new List<CompareRow>();
                    CompareKpis = new List<CompareKpi>();
                    return;
                }

                var o = JObject.Parse(text);
                var kpis = new List<CompareKpi>();
                foreach (var k in o["kpis"] ?? new JArray())
                {
                    kpis.Add(new CompareKpi
                    {
                        Key = (string)k["key"],
                        Label = (string)k["label"],
                        LowerIsBetter = (bool?)k["lower_is_better"] ?? true,
                    });
                }

                var rows = new List<CompareRow>();
                foreach (var r in o["rows"] ?? new JArray())
                {
                    var row = new CompareRow
                    {
                        Label = (string)r["label"],
                        IsOurs = (bool?)r["is_ours"] ?? false,
                        IsUserModel = (bool?)r["is_user_model"] ?? false,
                        Episodes = (int?)r["n_episodes"] ?? 0,
                        GridlockRate = (double?)r["gridlock_rate"] ?? 0d,
                    };
                    foreach (var k in kpis) row.Values[k.Key] = (double?)r[k.Key];
                    rows.Add(row);
                }

                CompareKpis = kpis;
                CompareRows = rows;
                CompareNote = (string)o["note"];
            }
            catch (Exception exc) { CompareError = exc.Message; }
        }

        // ------------------------------------------------------------------ helpers

        private static List<float> Rewards(JToken curve)
        {
            var list = new List<float>();
            if (curve == null) return list;
            foreach (var point in curve) list.Add((float?)point["reward"] ?? 0f);
            return list;
        }

        private static List<string> ToStrings(JToken token)
        {
            var list = new List<string>();
            if (token == null) return list;
            foreach (var t in token) list.Add((string)t);
            return list;
        }

        private static string Detail(string body)
        {
            try { return (string)JObject.Parse(body)["detail"] ?? body; }
            catch (Exception) { return body; }
        }
    }
}
