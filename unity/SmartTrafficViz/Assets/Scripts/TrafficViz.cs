// T-05-03 - the Unity client: renders a live SUMO episode from the same 1 Hz ws/unity feed the
// dashboard uses.
//
// Drop this one component into an empty scene and press Play; everything else (camera, light,
// roads, signal heads, vehicle pool) is created here. See unity/README.md.
//
// Interpolation is the FIXED-ENDPOINT form the T-05-03 DoD requires:
//
//     position = Vector3.Lerp(previousFrame, latestFrame, elapsed / frameInterval)
//
// and deliberately NOT the ease-out `Lerp(current, target, k * deltaTime)` written in
// notes/03-simulation.md 6.5 - that form is framerate-dependent (it converges at a different
// rate at 30 fps than at 144 fps) and never actually reaches the target. The audit caught that;
// the correction is recorded on backlog T-05-03 / orphan T-U05.
//
// The interval is MEASURED between arrivals rather than assumed to be 1 s, because the hub's
// `speed` control (src/api/live.py) lets an episode run at 1x, 5x or unpaced - so the real gap
// between frames is a property of the session, not a constant.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public class TrafficViz : MonoBehaviour
    {
        [Header("Hub")]
        [Tooltip("ws/unity endpoint of the FastAPI hub.")]
        public string Url = "ws://127.0.0.1:8000/ws/unity";

        [Header("Interpolation")]
        [Tooltip("Fallback gap between frames before two have been seen, in seconds.")]
        public float DefaultInterval = 1.0f;

        [Tooltip("Clamp on the measured interval, so one stalled frame cannot freeze the scene.")]
        public float MaxInterval = 3.0f;

        private sealed class Car
        {
            public GameObject Go;
            public Vector3 From, To;
            public float FromYaw, ToYaw;
            public bool Seen;
        }

        private SumoSocket _socket;
        private readonly Dictionary<string, Car> _cars = new Dictionary<string, Car>();
        private readonly Stack<GameObject> _pool = new Stack<GameObject>();
        private Dictionary<string, Renderer> _heads;
        private Transform _carRoot;

        private float _frameAt;         // Time.time when the latest frame was applied
        private float _interval;        // measured seconds between the last two frames
        private SimFrame _latest;
        private Material _carMaterial;

        private void Start()
        {
            var scene = IntersectionScene.BuildStatic();
            _heads = IntersectionScene.BuildSignalHeads(scene.transform);
            _carRoot = new GameObject("Vehicles").transform;

            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            _carMaterial = new Material(shader) { color = new Color(0.85f, 0.87f, 0.92f) };

            EnsureCamera();
            EnsureLight();

            _interval = DefaultInterval;
            _socket = new SumoSocket(Url);
            _socket.Start();
        }

        private void Update()
        {
            // Drain everything queued this tick: only the newest frame defines the target, but
            // each one must be consumed or the bounded queue would just refill.
            while (_socket.TryDequeue(out var frame)) Apply(frame);

            if (_latest == null) return;

            var t = _interval > 0f ? Mathf.Clamp01((Time.time - _frameAt) / _interval) : 1f;
            foreach (var car in _cars.Values)
            {
                car.Go.transform.position = Vector3.Lerp(car.From, car.To, t);
                var yaw = Mathf.LerpAngle(car.FromYaw, car.ToYaw, t);
                car.Go.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            }
        }

        /// <summary>Promotes a newly arrived frame to the interpolation target.</summary>
        private void Apply(SimFrame frame)
        {
            var now = Time.time;
            if (_latest != null)
            {
                var measured = now - _frameAt;
                if (measured > 0.001f) _interval = Mathf.Min(measured, MaxInterval);
            }
            _frameAt = now;
            _latest = frame;

            foreach (var car in _cars.Values) car.Seen = false;

            foreach (var v in frame.Payload.Vehicles)
            {
                var target = new Vector3(v.X, 0.75f, v.Y); // SUMO (x, y) -> Unity (x, z)
                if (!_cars.TryGetValue(v.Id, out var car))
                {
                    car = new Car { Go = Rent(), From = target, FromYaw = v.Angle };
                    _cars[v.Id] = car;
                }
                else
                {
                    // Start this leg where the last one ended, not from the drawn position: the
                    // endpoints must stay fixed for the whole interval or the motion eases out.
                    car.From = car.To;
                    car.FromYaw = car.ToYaw;
                }
                car.To = target;
                car.ToYaw = v.Angle;
                car.Seen = true;
            }

            // Anything absent from this frame has left the network - return it to the pool.
            var gone = new List<string>();
            foreach (var kv in _cars)
            {
                if (!kv.Value.Seen) gone.Add(kv.Key);
            }
            foreach (var id in gone)
            {
                Release(_cars[id].Go);
                _cars.Remove(id);
            }

            var colors = frame.Payload.Signal?.SignalColors;
            if (colors != null)
            {
                foreach (var kv in colors)
                {
                    if (_heads.TryGetValue(kv.Key, out var head))
                    {
                        head.material.color = IntersectionScene.ColorFor(kv.Value);
                    }
                }
            }
        }

        private GameObject Rent()
        {
            if (_pool.Count > 0)
            {
                var reused = _pool.Pop();
                reused.SetActive(true);
                return reused;
            }
            var car = GameObject.CreatePrimitive(PrimitiveType.Cube);
            car.name = "Vehicle";
            car.transform.SetParent(_carRoot);
            car.transform.localScale = new Vector3(1.8f, 1.5f, 4.5f); // passenger car, metres
            Destroy(car.GetComponent<BoxCollider>());                 // nothing here is physical
            car.GetComponent<Renderer>().sharedMaterial = _carMaterial;
            return car;
        }

        private void Release(GameObject car)
        {
            car.SetActive(false);
            _pool.Push(car);
        }

        private void EnsureCamera()
        {
            var cam = Camera.main;
            if (cam == null)
            {
                var go = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = go.AddComponent<Camera>();
            }
            cam.transform.position = new Vector3(0f, 95f, -95f);
            cam.transform.rotation = Quaternion.Euler(45f, 0f, 0f);
            cam.farClipPlane = 800f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.07f, 0.09f);
        }

        private void EnsureLight()
        {
            // FindFirstObjectByType, not the FindObjectOfType overload it replaced - the latter is
            // obsolete from Unity 2023 on and warns under Unity 6.
            if (FindFirstObjectByType<Light>() != null) return;
            var go = new GameObject("Directional Light");
            var light = go.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            go.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        private void OnGUI()
        {
            var status = _socket == null ? "no socket"
                : _socket.Connected ? "connected"
                : "connecting" + (_socket.LastError != null ? " (" + _socket.LastError + ")" : "");

            var simTime = _latest != null ? _latest.SimTime.ToString("F0") + " s" : "-";
            var phase = _latest?.Payload?.Signal != null ? "P" + _latest.Payload.Signal.PhaseIndex : "-";

            GUI.Label(new Rect(12, 8, 900, 22), "ws/unity: " + status + "    " + Url);
            GUI.Label(new Rect(12, 28, 900, 22),
                "sim time " + simTime + "    phase " + phase + "    vehicles " + _cars.Count);
            GUI.Label(new Rect(12, 48, 900, 22),
                "frames " + (_socket?.Received ?? 0) + "    dropped " + (_socket?.Dropped ?? 0) +
                "    interval " + _interval.ToString("F2") + " s");
        }

        private void OnDestroy()
        {
            _socket?.Dispose();
        }
    }
}
