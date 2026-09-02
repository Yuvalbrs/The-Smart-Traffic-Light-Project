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

        /// <summary>Latest hub session state, pushed in by AppShell's one-a-second poll.
        ///
        /// The socket stays happily connected after an episode ends, so the 3-D view simply froze
        /// on its last frame - visually identical to a stalled connection, with the HUD still
        /// reading "connected". A frozen picture the viewer cannot explain is the one thing a live
        /// demo cannot afford, so the HUD now says WHY it stopped.</summary>
        public string SessionState = "unknown";

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
        private Material _glass, _tyre, _trim, _headlight, _tail, _taxiPaint, _taxiSign;
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
            // One resolution path for the whole app (IntersectionScene.ResolveShader) - a second
            // copy of Shader.Find here is a second thing to fix when a build ships no shader.
            var shader = IntersectionScene.LitShader;
            var palette = new[]
            {
                new Color(0.87f, 0.88f, 0.90f), new Color(0.15f, 0.17f, 0.20f),
                new Color(0.75f, 0.20f, 0.18f), new Color(0.20f, 0.38f, 0.72f),
                new Color(0.90f, 0.72f, 0.20f), new Color(0.35f, 0.55f, 0.40f),
                new Color(0.55f, 0.57f, 0.62f),
            };
            _paints = new Material[palette.Length];
            if (shader != null)
            {
                for (var i = 0; i < palette.Length; i++)
                {
                    _paints[i] = new Material(shader) { color = palette[i] };
                }
                _glass = new Material(shader) { color = new Color(0.10f, 0.14f, 0.18f) };
                _tyre = new Material(shader) { color = new Color(0.07f, 0.07f, 0.08f) };
                _trim = new Material(shader) { color = new Color(0.13f, 0.13f, 0.15f) };
                // Lamps are emissive for the same reason the signal aspects are: at this camera
                // range an unlit dark-red face is indistinguishable from bodywork, and which end
                // of a car is which is exactly what the viewer is trying to read.
                _headlight = Emissive(shader, new Color(1.00f, 0.96f, 0.82f), 1.6f);
                _tail = Emissive(shader, new Color(0.85f, 0.12f, 0.10f), 1.9f);
                _taxiPaint = new Material(shader) { color = new Color(0.96f, 0.76f, 0.09f) };
                // Lit, so the roof sign still reads at night - which is when a taxi is easiest
                // to pick out of traffic and hardest to see by paint alone.
                _taxiSign = Emissive(shader, new Color(0.98f, 0.94f, 0.80f), 1.1f);
            }

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
            // Start() may have failed before the socket existed. Without this guard that costs one
            // NullReferenceException PER FRAME, forever - the symptom that buried the real error
            // under 66,000 identical stack traces in the player log.
            if (_socket == null) return;

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
                    if (!_heads.TryGetValue(kv.Key, out var head)) continue;
                    head.Set(AspectFor(kv.Key, kv.Value, colors));
                }
            }
        }


        /// <summary>The rightmost lane of every approach is SHARED, and its head must say so.
        ///
        /// `config/network/intersection.con.xml` gives lane 0 TWO connections - a through and a
        /// right - and `link_index_binding.yaml` binds the through half to the THROUGH movement,
        /// not the right one. The right half is permissive: link 0 is lowercase 'g' in all eight
        /// phases, i.e. green-but-give-way, always.
        ///
        /// So a head driven by the right movement alone is green 100% of the time, including
        /// through the six phases of eight where that lane's through traffic is held at red. On
        /// screen that is a green light above a stationary queue - which is what it looked like,
        /// and it was reported as a simulation bug. The simulation was right: the car under the
        /// lamp was through-bound and correctly stopped.
        ///
        /// One lamp cannot honestly show two independently controlled movements, so it shows the
        /// one that decides whether the lane can discharge at all: the more restrictive of the
        /// pair. Fixed here rather than in the network on purpose - splitting lane 0 into a
        /// right-turn-only lane would change capacity and invalidate the evaluation campaign.</summary>
        private static readonly Dictionary<string, string> SharedLaneThrough =
            new Dictionary<string, string>
            {
                { "M2", "M1" },    // N right shares lane 0 with N through
                { "M5", "M4" },    // E
                { "M8", "M7" },    // S
                { "M11", "M10" },  // W
            };

        private static string AspectFor(
            string movement, string own, Dictionary<string, string> all)
        {
            if (!SharedLaneThrough.TryGetValue(movement, out var partner)) return own;
            return all.TryGetValue(partner, out var other) ? MoreRestrictive(own, other) : own;
        }

        private static string MoreRestrictive(string a, string b) => Severity(a) >= Severity(b) ? a : b;

        /// <summary>red beats yellow beats green - anything unrecognised is treated as stop.</summary>
        private static int Severity(string aspect)
        {
            switch (aspect)
            {
                case "green": return 0;
                case "yellow": return 1;
                default: return 2;
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

            // A body, a cabin, glass, four wheels and lamps at both ends. The three-box version
            // read as a bar of soap: with no wheels a vehicle appears to hover, and hovering is
            // the single thing that most makes a traffic scene look like a toy. The wheels also
            // give the eye something at ground level to judge contact and scale against.
            //
            // Still boxes and cylinders, still no imported meshes - the parts are just placed
            // where a car has them. All of it is built once per pooled vehicle and never touched
            // again; the root carries no mesh so position/rotation stay on one transform.
            var car = new GameObject("Vehicle");
            car.transform.SetParent(_carRoot);

            var t = car.transform;
            // Body styles cycle rather than being drawn at random, so a queue is always visibly
            // mixed instead of occasionally coming out as six identical saloons.
            var style = _spawned % 4;
            var paint = style == 1 ? _taxiPaint : _paints[_spawned % _paints.Length];
            _spawned++;

            // +z is forward (see the yaw applied in Apply), so the cabin sits behind centre and
            // the windscreen faces +z.
            //
            // Four silhouettes off one parameterised shell: saloon, taxi, 4x4 and hatchback. All
            // the same road vehicle class - no buses or trucks, which SUMO is not routing here and
            // which would misrepresent the demand the agent is actually controlling.
            float bodyH, cabinH, cabinZ, cabinLen, ride, wheelR, len;
            switch (style)
            {
                case 1:  // taxi - saloon shell, longer cabin, and a roof sign
                    bodyH = 0.62f; cabinH = 0.64f; cabinZ = -0.24f; cabinLen = 2.3f;
                    ride = 0.05f; wheelR = 0.62f; len = 4.3f; break;
                case 2:  // 4x4 - taller body, taller glasshouse, bigger wheels, more ground clearance
                    bodyH = 0.86f; cabinH = 0.80f; cabinZ = -0.10f; cabinLen = 2.5f;
                    ride = 0.24f; wheelR = 0.76f; len = 4.5f; break;
                case 3:  // hatchback - short, stubby, cabin pushed back
                    bodyH = 0.60f; cabinH = 0.66f; cabinZ = -0.44f; cabinLen = 1.9f;
                    ride = 0.02f; wheelR = 0.58f; len = 3.7f; break;
                default: // saloon
                    bodyH = 0.62f; cabinH = 0.62f; cabinZ = -0.30f; cabinLen = 2.1f;
                    ride = 0.05f; wheelR = 0.62f; len = 4.3f; break;
            }

            var half = len / 2f;
            var cabinY = ride + bodyH / 2f + cabinH / 2f - 0.01f;

            AddPart(t, "Body", new Vector3(1.8f, bodyH, len), new Vector3(0f, ride, 0f), paint);
            AddPart(t, "Skirt", new Vector3(1.66f, 0.30f, len - 0.3f), new Vector3(0f, ride - 0.31f, 0f), _trim);
            AddPart(t, "Cabin", new Vector3(1.58f, cabinH, cabinLen), new Vector3(0f, cabinY, cabinZ), paint);

            var glassH = cabinH * 0.78f;
            AddPart(t, "Windscreen", new Vector3(1.50f, glassH, 0.14f),
                new Vector3(0f, cabinY, cabinZ + cabinLen / 2f), _glass);
            AddPart(t, "Rear glass", new Vector3(1.50f, glassH * 0.9f, 0.14f),
                new Vector3(0f, cabinY, cabinZ - cabinLen / 2f), _glass);
            AddPart(t, "Side glass L", new Vector3(0.10f, glassH, cabinLen - 0.4f),
                new Vector3(-0.79f, cabinY, cabinZ), _glass);
            AddPart(t, "Side glass R", new Vector3(0.10f, glassH, cabinLen - 0.4f),
                new Vector3(0.79f, cabinY, cabinZ), _glass);

            // Lamps: warm at the front, red at the back. Two small faces are enough to tell which
            // way a car is pointing from the overhead camera, which the plain boxes could not.
            AddPart(t, "Headlight L", new Vector3(0.42f, 0.20f, 0.10f), new Vector3(-0.62f, ride + 0.05f, half), _headlight);
            AddPart(t, "Headlight R", new Vector3(0.42f, 0.20f, 0.10f), new Vector3(0.62f, ride + 0.05f, half), _headlight);
            AddPart(t, "Tail L", new Vector3(0.40f, 0.18f, 0.10f), new Vector3(-0.63f, ride + 0.09f, -half), _tail);
            AddPart(t, "Tail R", new Vector3(0.40f, 0.18f, 0.10f), new Vector3(0.63f, ride + 0.09f, -half), _tail);

            if (style == 1)
            {
                // The roof sign is what actually makes a taxi readable from above - the paint
                // colour alone is just another yellow car.
                AddPart(t, "Taxi sign", new Vector3(0.75f, 0.24f, 0.30f),
                    new Vector3(0f, cabinY + cabinH / 2f + 0.12f, cabinZ + 0.2f), _taxiSign);
            }
            if (style == 2)
            {
                AddPart(t, "Roof rack", new Vector3(1.30f, 0.10f, cabinLen - 0.5f),
                    new Vector3(0f, cabinY + cabinH / 2f + 0.06f, cabinZ), _trim);
            }

            var axle = half - wheelR - 0.28f;
            var wheelY = ride - bodyH / 2f + wheelR * 0.35f;
            AddWheel(t, "Wheel FL", new Vector3(-0.86f, wheelY, axle), wheelR);
            AddWheel(t, "Wheel FR", new Vector3(0.86f, wheelY, axle), wheelR);
            AddWheel(t, "Wheel RL", new Vector3(-0.86f, wheelY, -axle), wheelR);
            AddWheel(t, "Wheel RR", new Vector3(0.86f, wheelY, -axle), wheelR);
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

        private static Material Emissive(Shader shader, Color c, float strength)
        {
            var m = new Material(shader) { color = c };
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", c * strength);
            return m;
        }

        /// <summary>A cylinder laid on its side, so its round face points out across the car.</summary>
        private void AddWheel(Transform parent, string name, Vector3 offset, float radius)
        {
            var wheel = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            wheel.name = name;
            wheel.transform.SetParent(parent);
            // Unity's cylinder is 2 units tall along +y, so y-scale is the HALF width.
            wheel.transform.localScale = new Vector3(radius, 0.12f, radius);
            wheel.transform.localPosition = offset;
            wheel.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
            Destroy(wheel.GetComponent<CapsuleCollider>());
            wheel.GetComponent<Renderer>().sharedMaterial = _tyre;
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

            // FindFirstObjectByType, not the FindObjectOfType overload it replaced - the latter is
            // obsolete from Unity 2023 on and warns under Unity 6.
            var existing = FindFirstObjectByType<Light>();
            if (existing != null)
            {
                // Still hand it over: the sun's colour, angle and the ambient level are all owned
                // by DayNight, and a scene rebuilt while night is on must come back at night.
                DayNight.RegisterSun(existing);
                return;
            }

            var go = new GameObject("Directional Light");
            var light = go.AddComponent<Light>();
            light.type = LightType.Directional;
            DayNight.RegisterSun(light);
        }

        private void OnGUI()
        {
            string status;
            if (_socket == null) status = "no socket";
            else if (!_socket.Connected)
                status = "connecting" + (_socket.LastError != null ? " (" + _socket.LastError + ")" : "");
            else
            {
                // Connected. Whether frames are still coming is a property of the SESSION, not the
                // socket - so name the session state rather than let a frozen scene read "connected".
                switch (SessionState)
                {
                    case "running":
                    case "starting":
                        status = "connected - streaming";
                        break;
                    case "finished":
                        status = "connected - episode finished (no more frames)";
                        break;
                    case "stopped":
                        status = "connected - episode stopped (no more frames)";
                        break;
                    case "failed":
                        status = "connected - episode FAILED (no more frames)";
                        break;
                    case "none":
                        status = "connected - no session running";
                        break;
                    default:
                        status = "connected";
                        break;
                }
            }

            var simTime = _latest != null ? _latest.SimTime.ToString("F0") + " s" : "-";
            var phase = _latest?.Payload?.Signal != null ? "P" + _latest.Payload.Signal.PhaseIndex : "-";

            // One panel with its own backdrop, laid out from a single origin below the shell's
            // top bar. The previous version scattered labels at hard-coded y values that
            // overlapped the bar and each other.
            const float top = AppShell.TopBarHeight + 8f;
            const float pad = 12f, line = 20f;
            var panel = new Rect(8f, top, 430f, line * 3f + 16f);
            UITheme.Backdrop(panel);

            var y = panel.y + 8f;
            GUI.Label(new Rect(panel.x + pad, y, panel.width - pad * 2f, line),
                "ws/unity: " + status, UITheme.Heading);
            y += line;
            GUI.Label(new Rect(panel.x + pad, y, panel.width - pad * 2f, line),
                "sim " + simTime + "     phase " + phase + "     vehicles " + _cars.Count,
                UITheme.Label);
            y += line;
            GUI.Label(new Rect(panel.x + pad, y, panel.width - pad * 2f, line),
                "frames " + (_socket?.Received ?? 0) + "   dropped " + (_socket?.Dropped ?? 0) +
                "   interval " + _interval.ToString("F2") + " s", UITheme.Hint);        }

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
