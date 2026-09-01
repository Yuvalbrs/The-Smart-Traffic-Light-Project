// The ws/dashboard subscriber - the numbers half of the application.
//
// The viewer already consumes ws/unity, which carries geometry: every vehicle's position and the
// signal colours. The dashboard channel carries the DERIVED per-second summary instead - queues
// and pressure per movement, the running KPI estimates, and the frozen LSTM's forecast when a
// hybrid controller is driving. Both are fan-outs of the same 1 Hz frames from the same episode
// (src/api/hub.py), which is what makes the picture and the numbers incapable of disagreeing.
//
// Deliberately a second socket rather than deriving these from sim_frame on this side: the hub
// already computes them once, and a second implementation here would be free to drift from the
// one the dashboard and the results tables use.
//
// Structure mirrors SumoSocket exactly, including the bounded queue: keep the newest frames and
// drop the oldest, so a stall reads as a jump rather than as growing lag.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace SmartTraffic
{
    /// <summary>One dashboard frame, mirrored from <c>src/api/wire.py::dashboard_frame</c>.</summary>
    public class DashboardFrame
    {
        [JsonProperty("type")] public string Type;
        [JsonProperty("schema_version")] public string SchemaVersion;
        [JsonProperty("seq")] public long Seq;
        [JsonProperty("episode_id")] public long EpisodeId;
        [JsonProperty("sim_time")] public double SimTime;
        [JsonProperty("current_phase")] public int CurrentPhase;
        [JsonProperty("last_action")] public int LastAction;

        /// <summary>Per-movement halting counts, M0..M11.</summary>
        [JsonProperty("queue_lengths")] public List<float> QueueLengths;

        /// <summary>Per-movement pressure, M0..M11, unnormalised.</summary>
        [JsonProperty("pressures")] public List<float> Pressures;

        [JsonProperty("running_kpis")] public RunningKpis RunningKpis;

        /// <summary>
        /// The frozen forecaster's 36 values (3 horizons x 12 movements), or null for every
        /// controller that carries no forecaster - which is most of them. Explicitly null rather
        /// than zero-filled, because zeros would read as a confident "no traffic" forecast.
        /// </summary>
        [JsonProperty("forecast_next_30s")] public List<float> ForecastNext30s;
    }

    public class RunningKpis
    {
        [JsonProperty("avg_wait_so_far")] public double AvgWaitSoFar;
        [JsonProperty("throughput_so_far")] public double ThroughputSoFar;
        [JsonProperty("current_queue_total")] public double CurrentQueueTotal;
    }

    public class DashboardFeed : IDisposable
    {
        private const int MaxQueued = 8;   // matches src/api/hub.py MAX_QUEUE

        private readonly ConcurrentQueue<DashboardFrame> _frames = new ConcurrentQueue<DashboardFrame>();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private ClientWebSocket _socket;

        public string Url { get; }
        public bool Connected { get; private set; }
        public string LastError { get; private set; }
        public long Received { get; private set; }
        public long Dropped { get; private set; }

        public DashboardFeed(string url)
        {
            Url = url;
        }

        public void Start()
        {
            Task.Run(() => RunAsync(_cts.Token));
        }

        /// <summary>Pops the next frame, or false when none is waiting. Main thread only.</summary>
        public bool TryDequeue(out DashboardFrame frame) => _frames.TryDequeue(out frame);

        private async Task RunAsync(CancellationToken token)
        {
            var buffer = new byte[1 << 16];
            while (!token.IsCancellationRequested)
            {
                try
                {
                    _socket = new ClientWebSocket();
                    await _socket.ConnectAsync(new Uri(Url), token);
                    Connected = true;
                    LastError = null;

                    var text = new StringBuilder();
                    while (_socket.State == WebSocketState.Open && !token.IsCancellationRequested)
                    {
                        text.Clear();
                        WebSocketReceiveResult result;
                        do
                        {
                            result = await _socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                            if (result.MessageType == WebSocketMessageType.Close) break;
                            text.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                        }
                        while (!result.EndOfMessage);

                        if (result.MessageType == WebSocketMessageType.Close) break;
                        Enqueue(text.ToString());
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    LastError = ex.Message;
                }

                Connected = false;
                SafeAbort();
                if (token.IsCancellationRequested) break;
                // The hub may simply have no session running yet; retry rather than give up.
                try { await Task.Delay(1500, token); } catch (OperationCanceledException) { break; }
            }
        }

        private void Enqueue(string json)
        {
            if (string.IsNullOrEmpty(json)) return;
            DashboardFrame frame;
            try
            {
                frame = JsonConvert.DeserializeObject<DashboardFrame>(json);
            }
            catch (JsonException ex)
            {
                LastError = ex.Message;   // the hello envelope and any malformed frame land here
                return;
            }
            // The channel's first message is a {"type":"hello"} envelope with no payload.
            if (frame == null || frame.RunningKpis == null) return;

            // One line, once, on the first successfully parsed frame. This is the only evidence
            // available that Newtonsoft's reflection into DashboardFrame survived the build: if
            // managed stripping ever removes it the fields come back as zeros with no exception,
            // which on screen is indistinguishable from a network fault. The player log makes that
            // failure visible without anyone having to squint at the UI.
            if (Received == 0)
            {
                UnityEngine.Debug.Log(
                    $"[dashboard-feed] first frame parsed OK: sim_time={frame.SimTime} " +
                    $"phase={frame.CurrentPhase} queues={(frame.QueueLengths == null ? -1 : frame.QueueLengths.Count)} " +
                    $"avg_wait={frame.RunningKpis.AvgWaitSoFar}");
            }

            _frames.Enqueue(frame);
            Received++;
            while (_frames.Count > MaxQueued && _frames.TryDequeue(out _)) Dropped++;
        }

        private void SafeAbort()
        {
            try { _socket?.Abort(); _socket?.Dispose(); }
            catch (Exception) { /* teardown is best effort */ }
            _socket = null;
        }

        public void Dispose()
        {
            _cts.Cancel();
            SafeAbort();
            _cts.Dispose();
        }
    }
}
