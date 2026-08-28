// T-05-03 - starting and stopping episodes from inside Unity.
//
// Talks to the same REST endpoints the React dashboard uses (src/api/server.py): POST /sessions,
// DELETE /sessions/current, GET /sessions/current. Unity is a client of the hub like any other -
// it never touches SUMO, so the picture can never disagree with what the dashboard reports.
//
// Only one episode can exist at a time, and that is forced rather than stylistic: libsumo runs
// in-process and holds a single global simulation, so a second concurrent episode would corrupt
// the first. Switching scenario or controller therefore means stop-then-start, which is what
// Switch() does - a plain POST while one runs correctly answers 409.
//
// Requests run on background tasks (HttpClient), with results parked in fields the main thread
// reads during OnGUI. Unity API calls are main-thread only, so nothing here touches the scene.

using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SmartTraffic
{
    public struct RunConfig
    {
        public string Controller;
        public string Scenario;
        public int Seed;
        public int EpisodeLengthS;
        public float Speed;

        public static RunConfig Default => new RunConfig
        {
            Controller = "webster",
            Scenario = "SCN-04",
            Seed = 7000,
            EpisodeLengthS = 3600,
            // 5x is the demo default: unpaced runs an hour of traffic in about five seconds.
            Speed = 5f,
        };
    }

    public class SessionControl
    {
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

        private readonly string _base;

        /// <summary>Busy while a start/stop is in flight, so the UI can disable its buttons.</summary>
        public bool Busy { get; private set; }
        public string LastError { get; private set; }
        public string State { get; private set; } = "unknown";
        public string RunningController { get; private set; } = "-";
        public string RunningScenario { get; private set; } = "-";
        public double SimTime { get; private set; }

        /// <param name="wsUrl">The ws/unity URL; the REST base is derived from it.</param>
        public SessionControl(string wsUrl)
        {
            // ws://127.0.0.1:8000/ws/unity -> http://127.0.0.1:8000
            var http = wsUrl.Replace("ws://", "http://").Replace("wss://", "https://");
            var cut = http.IndexOf("/ws/", StringComparison.Ordinal);
            _base = cut > 0 ? http.Substring(0, cut) : http.TrimEnd('/');
        }

        public async void Switch(RunConfig cfg)
        {
            if (Busy) return;
            Busy = true;
            LastError = null;
            try
            {
                // Stop whatever is running first. 404 simply means nothing was.
                await Http.DeleteAsync(_base + "/sessions/current");

                var body = JsonConvert.SerializeObject(new
                {
                    controller = cfg.Controller,
                    scenario = cfg.Scenario,
                    seed = cfg.Seed,
                    episode_length_s = cfg.EpisodeLengthS,
                    trace = true,
                    speed = cfg.Speed,
                });
                var resp = await Http.PostAsync(_base + "/sessions",
                    new StringContent(body, Encoding.UTF8, "application/json"));
                var text = await resp.Content.ReadAsStringAsync();
                if (!resp.IsSuccessStatusCode)
                {
                    LastError = (int)resp.StatusCode + ": " + Detail(text);
                    return;
                }
                Absorb(text);
            }
            catch (Exception exc)
            {
                LastError = exc.Message;
            }
            finally
            {
                Busy = false;
            }
        }

        public async void Stop()
        {
            if (Busy) return;
            Busy = true;
            LastError = null;
            try
            {
                await Http.DeleteAsync(_base + "/sessions/current");
                State = "stopped";
            }
            catch (Exception exc)
            {
                LastError = exc.Message;
            }
            finally
            {
                Busy = false;
            }
        }

        /// <summary>Polls session status. Cheap enough to call about once a second.</summary>
        public async void Refresh()
        {
            try
            {
                var resp = await Http.GetAsync(_base + "/sessions/current");
                if (!resp.IsSuccessStatusCode) { State = "none"; return; }
                Absorb(await resp.Content.ReadAsStringAsync());
            }
            catch (Exception exc)
            {
                LastError = exc.Message;
            }
        }

        private void Absorb(string json)
        {
            var o = JObject.Parse(json);
            State = (string)o["state"] ?? "unknown";
            RunningController = (string)o["controller"] ?? "-";
            RunningScenario = (string)o["scenario"] ?? "-";
            SimTime = (double?)o["sim_time"] ?? 0d;
        }

        private static string Detail(string body)
        {
            try { return (string)JObject.Parse(body)["detail"] ?? body; }
            catch (Exception) { return body; }
        }
    }
}
