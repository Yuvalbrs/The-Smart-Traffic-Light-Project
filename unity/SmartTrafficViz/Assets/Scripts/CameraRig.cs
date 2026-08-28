// T-05-03 - camera control for the Game view.
//
// The Game view is locked to Main Camera, so without this the only way to look around is the
// editor's Scene tab - not available in a build, and awkward to drive in front of an audience.
// The demo needs to pull back for the whole network, push in on the junction to show a queue
// forming, and sit at a driver's eye level down one approach, so that has to be a runtime control.
//
// Two modes. Orbit is a target-and-distance rig, which is what preset framings need. Free is a
// fly camera for exploring - deliberately separate, because an orbit rig cannot leave its target
// and a fly camera cannot snap to a framing.
//
// Legacy Input (ProjectSettings activeInputHandler: 0) rather than the Input System package: no
// extra dependency for what is five axes of camera movement.

using UnityEngine;

namespace SmartTraffic
{
    [RequireComponent(typeof(Camera))]
    public class CameraRig : MonoBehaviour
    {
        public enum Mode { Orbit, Free }

        private struct Preset
        {
            public string Name;
            public Vector3 Target;
            public float Distance, Yaw, Pitch;

            public Preset(string name, Vector3 target, float distance, float yaw, float pitch)
            {
                Name = name; Target = target; Distance = distance; Yaw = yaw; Pitch = pitch;
            }
        }

        // Yaw 0 puts the camera SOUTH of its target looking north, so an approach is framed by
        // standing beyond it and looking back at the junction: north approach -> yaw 180.
        private static readonly Preset[] Presets =
        {
            new Preset("Network",  Vector3.zero,             283f,   0f, 45f),
            new Preset("Junction", Vector3.zero,              70f,   0f, 35f),
            new Preset("Top down", Vector3.zero,             210f,   0f, 89f),
            new Preset("North",    new Vector3(0f, 0f, 48f),  85f, 180f, 14f),
            new Preset("East",     new Vector3(48f, 0f, 0f),  85f, 270f, 14f),
            new Preset("South",    new Vector3(0f, 0f, -48f), 85f,   0f, 14f),
            new Preset("West",     new Vector3(-48f, 0f, 0f), 85f,  90f, 14f),
        };

        [Header("Orbit state")]
        public Vector3 Target = Vector3.zero;
        public float Distance = 283f;
        public float Yaw;
        public float Pitch = 45f;

        [Header("Limits")]
        public float MinDistance = 15f;
        public float MaxDistance = 700f;
        public float MinPitch = 4f;
        public float MaxPitch = 89f;

        [Header("Sensitivity")]
        public float ZoomSpeed = 1.2f;
        public float OrbitSpeed = 4f;
        public float PanSpeed = 0.6f;
        public float FlySpeed = 60f;

        public Mode Current { get; private set; } = Mode.Orbit;

        private float _freeYaw, _freePitch;

        private void Start() => ApplyOrbit();

        private void LateUpdate()
        {
            if (Current == Mode.Free) UpdateFree();
            else UpdateOrbit();
        }

        // --- orbit ---

        private void UpdateOrbit()
        {
            // Zoom proportional to distance: one notch should feel the same 500 m out and 30 m in,
            // which a fixed metres-per-notch step does not.
            var wheel = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(wheel) > 0.0001f)
            {
                Distance = Mathf.Clamp(Distance * (1f - wheel * ZoomSpeed), MinDistance, MaxDistance);
            }

            if (Input.GetMouseButton(1))
            {
                Yaw += Input.GetAxis("Mouse X") * OrbitSpeed;
                Pitch = Mathf.Clamp(Pitch - Input.GetAxis("Mouse Y") * OrbitSpeed, MinPitch, MaxPitch);
            }

            if (Input.GetMouseButton(2))
            {
                var scale = PanSpeed * Distance / 283f;
                var right = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;
                var fwd = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
                Target -= (right * Input.GetAxis("Mouse X") + fwd * Input.GetAxis("Mouse Y")) * scale;
            }

            for (var i = 0; i < Presets.Length && i < 9; i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1 + i)) Use(i);
            }
            if (Input.GetKeyDown(KeyCode.F)) SetMode(Mode.Free);

            ApplyOrbit();
        }

        private void ApplyOrbit()
        {
            var rotation = Quaternion.Euler(Pitch, Yaw, 0f);
            transform.rotation = rotation;
            transform.position = Target - rotation * Vector3.forward * Distance;
        }

        // --- free fly ---

        private void UpdateFree()
        {
            if (Input.GetMouseButton(1))
            {
                _freeYaw += Input.GetAxis("Mouse X") * OrbitSpeed;
                _freePitch = Mathf.Clamp(_freePitch - Input.GetAxis("Mouse Y") * OrbitSpeed, -89f, 89f);
                transform.rotation = Quaternion.Euler(_freePitch, _freeYaw, 0f);
            }

            var speed = FlySpeed * (Input.GetKey(KeyCode.LeftShift) ? 4f : 1f) * Time.deltaTime;
            var move = Vector3.zero;
            if (Input.GetKey(KeyCode.W)) move += transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= transform.forward;
            if (Input.GetKey(KeyCode.D)) move += transform.right;
            if (Input.GetKey(KeyCode.A)) move -= transform.right;
            if (Input.GetKey(KeyCode.E)) move += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) move -= Vector3.up;
            transform.position += move.normalized * speed;

            if (Input.GetKeyDown(KeyCode.Escape)) SetMode(Mode.Orbit);
            for (var i = 0; i < Presets.Length && i < 9; i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1 + i)) { SetMode(Mode.Orbit); Use(i); }
            }
        }

        private void SetMode(Mode mode)
        {
            if (mode == Mode.Free)
            {
                // Carry the current view across rather than snapping, so switching to Free feels
                // like taking hold of the camera you were already looking through.
                var e = transform.rotation.eulerAngles;
                _freePitch = e.x > 180f ? e.x - 360f : e.x;
                _freeYaw = e.y;
            }
            Current = mode;
        }

        private void Use(int index)
        {
            var p = Presets[index];
            Target = p.Target; Distance = p.Distance; Yaw = p.Yaw; Pitch = p.Pitch;
            Current = Mode.Orbit;
            ApplyOrbit();
        }

        private void OnGUI()
        {
            const float w = 96f, h = 30f, pad = 8f, barH = 74f;

            var free = Current == Mode.Free;
            var bar = new Rect(0f, Screen.height - barH, Screen.width, barH);
            UITheme.Backdrop(bar);

            GUI.Label(new Rect(pad + 6f, bar.y + 5f, 120f, 18f), "CAMERA", UITheme.Heading);

            var hint = free
                ? "WASD move   Q/E down/up   shift faster   right-drag look   Esc orbit"
                : "wheel zoom   right-drag orbit   middle-drag pan";
            GUI.Label(new Rect(pad + 90f, bar.y + 6f, Screen.width - pad - 100f, 18f),
                hint, UITheme.Hint);

            var x = pad + 6f;
            var y = bar.y + 30f;
            for (var i = 0; i < Presets.Length; i++)
            {
                var active = !free && Mathf.Approximately(Yaw, Presets[i].Yaw)
                                   && Mathf.Approximately(Distance, Presets[i].Distance);
                var style = active ? UITheme.ButtonOn : UITheme.Button;
                if (GUI.Button(new Rect(x, y, w, h), Presets[i].Name + "  " + (i + 1), style)) Use(i);
                x += w + pad;
            }

            x += 8f;
            if (GUI.Button(new Rect(x, y, w + 34f, h), free ? "Free camera  ON" : "Free camera  F",
                    free ? UITheme.ButtonOn : UITheme.Button))
            {
                SetMode(free ? Mode.Orbit : Mode.Free);
            }
        }
    }
}
