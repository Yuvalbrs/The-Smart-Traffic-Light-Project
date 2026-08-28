// T-05-03 - the ws/unity subscriber.
//
// Uses System.Net.WebSockets.ClientWebSocket (built into Unity's .NET Standard profile) rather
// than the NativeWebSocket package named in the backlog DoD. NativeWebSocket exists to work
// around WebGL's lack of System.Net sockets; this is a desktop demo, so the built-in client is
// one fewer git-URL package to resolve on a deadline. Documented deviation.
//
// The socket runs on a background task and hands frames to the main thread through a bounded
// queue - Unity API calls are main-thread only, and the hub already drops frames rather than
// blocking SUMO, so a slow renderer must never push back on the network read.

using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace SmartTraffic
{
    public class SumoSocket : IDisposable
    {
        // Matches the hub's own per-client queue (src/api/hub.py MAX_QUEUE): keep the newest
        // frames and drop the oldest, so a stall shows up as a jump, never as growing lag.
        private const int MaxQueued = 8;

        private readonly ConcurrentQueue<SimFrame> _frames = new ConcurrentQueue<SimFrame>();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private ClientWebSocket _socket;

        public string Url { get; }
        public bool Connected { get; private set; }
        public string LastError { get; private set; }
        public long Received { get; private set; }
        public long Dropped { get; private set; }

        public SumoSocket(string url)
        {
            Url = url;
        }

        public void Start()
        {
            Task.Run(() => RunAsync(_cts.Token));
        }

        /// <summary>Pops the next frame, or returns false when none is waiting. Main thread only.</summary>
        public bool TryDequeue(out SimFrame frame) => _frames.TryDequeue(out frame);

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
                            if (result.MessageType == WebSocketMessageType.Close)
                            {
                                break;
                            }
                            text.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                        } while (!result.EndOfMessage);

                        if (result.MessageType == WebSocketMessageType.Close) break;
                        if (text.Length == 0) continue;

                        var frame = JsonConvert.DeserializeObject<SimFrame>(text.ToString());
                        // The hub greets every subscriber with a {"type":"hello"} envelope that
                        // carries no payload - skip it rather than null-checking downstream.
                        if (frame?.Payload == null || frame.Type != "sim_frame") continue;

                        Received++;
                        _frames.Enqueue(frame);
                        while (_frames.Count > MaxQueued && _frames.TryDequeue(out _)) Dropped++;
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception exc)
                {
                    LastError = exc.Message;
                }

                Connected = false;
                SafeAbort();
                if (token.IsCancellationRequested) break;
                // The hub may simply have no session running yet; retry rather than give up.
                try { await Task.Delay(1500, token); } catch (OperationCanceledException) { break; }
            }
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
