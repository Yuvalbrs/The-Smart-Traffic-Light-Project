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
        private Dictionary<string, IntersectionScene.SignalHead> _heads;
        private Transform _carRoot;

        private float _frameAt;         // Time.time when the latest frame was applied
        private float _interval;        // measured seconds between the last two frames
        private SimFrame _latest;
        private Material[] _paints;
        private Material _glass;
        private int _spawned;
        private GameObject _sceneRoot;
        private CameraRig _rig;

        private void Start()
        {
            _sceneRoot = IntersectionScene.BuildStatic();
            _heads = IntersectionScene.BuildSignalHeads(_sceneRoot.transform);
            Scenery.Build(_sceneRoot.transform);
            _carRoot = new GameObject("Vehicles").transform;

            // A fixed palette shared by sharedMaterial, not one material per car: cars are pooled
            // and recycled, so per-instance materials would leak a new one on every spawn.
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var palette = new[]
            {
                new Color(0.87f, 0.88f, 0.90f), new Color(0.15f, 0.17f, 0.20f),
                new Color(0.75f, 0.20f, 0.18f), new Color(0.20f, 0.38f, 0.72f),
                new Color(0.90f, 0.72f, 0.20f), new Color(0.35f, 0.55f, 0.40f),
                new Color(0.55f, 0.57f, 0.62f),
            };
            _paints = new Material[palette.Length];
            for (var i = 0; i < palette.Length; i++) _paints[i] = new Material(shader) { color = palette[i] };
            _glass = new Material(shader) { color = new Color(0.10f, 0.14f, 0.18f) };

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
                // SUMO (x, y) -> Unity (x, z). The y lift puts the body shell just above the road
                // surface: the body is 0.85 tall and centred on the root.
                var target = new Vector3(v.X, 0.52f, v.Y);
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
                    if (_heads.TryGetValue(kv.Key, out var head)) head.Set(kv.Value);
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

            // A body plus a smaller cabin reads as a car at this camera range, where a single box
            // reads as a brick. The root carries no mesh so position/rotation stay on one
            // transform and the parts never need touching again.
            var car = new GameObject("Vehicle");
            car.transform.SetParent(_carRoot);

            var paint = _paints[_spawned++ % _paints.Length];
            AddPart(car.transform, "Body", new Vector3(1.8f, 0.85f, 4.3f), new Vector3(0f, 0f, 0f), paint);
            AddPart(car.transform, "Cabin", new Vector3(1.55f, 0.7f, 2.0f), new Vector3(0f, 0.72f, -0.25f), paint);
            AddPart(car.transform, "Glass", new Vector3(1.58f, 0.42f, 0.12f), new Vector3(0f, 0.78f, 0.76f), _glass);
            return car;
        }

        private static void AddPart(Transform parent, string name, Vector3 size, Vector3 offset, Material mat)
        {
            var part = GameObject.CreatePrimitive(PrimitiveType.Cube);
            part.name = name;
            part.transform.SetParent(parent);
            part.transform.localScale = size;
            part.transform.localPosition = offset;
            Destroy(part.GetComponent<BoxCollider>()); // nothing here is physical
            part.GetComponent<Renderer>().sharedMaterial = mat;
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
            cam.farClipPlane = 1500f;
            // Skybox when the project has one (it does by default), else a daylight-ish solid so
            // the generated hills and clouds still sit against sky rather than a black void.
            cam.clearFlags = RenderSettings.skybox != null
                ? CameraClearFlags.Skybox
                : CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.53f, 0.70f, 0.87f);

            // Framing and all camera movement belong to the rig; it sets the transform every
            // LateUpdate, so anything written here would be overwritten on the first frame.
            _rig = cam.GetComponent<CameraRig>();
            if (_rig == null) _rig = cam.gameObject.AddComponent<CameraRig>();
        }

        private void EnsureLight()
        {
            // Flat ambient on top of the key light: without it the unlit faces of the cars and
            // signal housings go almost black at this camera angle and the scene reads as mush.
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.42f, 0.45f, 0.50f);

            // FindFirstObjectByType, not the FindObjectOfType overload it replaced - the latter is
            // obsolete from Unity 2023 on and warns under Unity 6.
            if (FindFirstObjectByType<Light>() != null) return;
            var go = new GameObject("Directional Light");
            var light = go.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.25f;
            go.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        private void OnGUI()
        {
            var status = _socket == null ? "no socket"
                : _socket.Connected ? "connected"
                : "connecting" + (_socket.LastError != null ? " (" + _socket.LastError + ")" : "");

            var simTime = _latest != null ? _latest.SimTime.ToString("F0") + " s" : "-";
            var phase = _latest?.Payload?.Signal != null ? "P" + _latest.Payload.Signal.PhaseIndex : "-";

            // Offset below the shell's top bar so the two do not overlap.
            GUI.Label(new Rect(12, 6, 900, 20), "ws/unity: " + status, UITheme.Heading);
            GUI.Label(new Rect(12, 38, 900, 20),
                "sim time " + simTime + "     phase " + phase + "     vehicles " + _cars.Count,
                UITheme.Label);
            GUI.Label(new Rect(12, 58, 900, 20),
                "frames " + (_socket?.Received ?? 0) + "     dropped " + (_socket?.Dropped ?? 0) +
                "     interval " + _interval.ToString("F2") + " s     " + Url,
                UITheme.Hint);        }

        private void OnDestroy()
        {
            _socket?.Dispose();

            // The scene and vehicle roots are separate top-level objects, not children of this
            // component, so returning to the menu has to remove them explicitly - and the camera
            // rig with them, or its control bar would keep drawing over the menu.
            if (_sceneRoot != null) Destroy(_sceneRoot);
            if (_carRoot != null) Destroy(_carRoot.gameObject);
            if (_rig != null) Destroy(_rig);
        }
    }
}
