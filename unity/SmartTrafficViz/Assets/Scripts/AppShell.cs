// T-05-03 - the app shell: a main menu and navigation between screens.
//
// "Screen" here is an application state, not a Unity scene asset. Real .unity scenes would each
// have to be authored in the editor, added to Build Settings, and committed as serialised YAML
// that cannot be reviewed in a diff - and every one of them would still need the same runtime
// code to build its contents, because the intersection is generated rather than authored. A
// state machine over one scene gives the same navigation with none of that.
//
// The 3-D view is created on entry and torn down on exit, so the menu is not sitting on top of a
// live simulation burning frames, and returning to it releases the WebSocket.

using UnityEngine;

namespace SmartTraffic
{
    public class AppShell : MonoBehaviour
    {
        public enum Screen { Menu, Live, Controls, About }

        public string HubUrl = "ws://127.0.0.1:8000/ws/unity";

        public Screen Current { get; private set; } = Screen.Menu;

        private TrafficViz _viz;
        private Camera _menuCam;

        private void Start()
        {
            EnsureMenuCamera();
        }

        private void Update()
        {
            // Escape backs out one level, from anywhere.
            if (Input.GetKeyDown(KeyCode.Escape) && Current != Screen.Menu && !InFreeCamera())
            {
                Go(Screen.Menu);
            }
        }

        private bool InFreeCamera()
        {
            // Free-fly owns Escape while it is active - it uses it to drop back to orbit.
            var rig = _viz != null ? Camera.main?.GetComponent<CameraRig>() : null;
            return rig != null && rig.Current == CameraRig.Mode.Free;
        }

        public void Go(Screen screen)
        {
            if (screen == Current) return;

            if (Current == Screen.Live && _viz != null)
            {
                Destroy(_viz.gameObject);
                _viz = null;
            }

            Current = screen;

            if (screen == Screen.Live)
            {
                var host = new GameObject("TrafficViz");
                _viz = host.AddComponent<TrafficViz>();
                _viz.Url = HubUrl;
            }
            EnsureMenuCamera();
        }

        /// <summary>A camera has to exist even with no scene, or the menu draws over a black void.</summary>
        private void EnsureMenuCamera()
        {
            if (Camera.main != null) { _menuCam = Camera.main; return; }
            var go = new GameObject("Main Camera") { tag = "MainCamera" };
            _menuCam = go.AddComponent<Camera>();
            _menuCam.clearFlags = CameraClearFlags.SolidColor;
            _menuCam.backgroundColor = new Color(0.05f, 0.07f, 0.10f);
            go.transform.position = new Vector3(0f, 60f, -120f);
            go.transform.rotation = Quaternion.Euler(20f, 0f, 0f);
        }

        private void OnGUI()
        {
            switch (Current)
            {
                case Screen.Menu: DrawMenu(); break;
                case Screen.Controls: DrawControls(); break;
                case Screen.About: DrawAbout(); break;
                case Screen.Live: DrawLiveBar(); break;
            }
        }

        private void DrawMenu()
        {
            const float w = 420f, rowH = 44f, gap = 10f;
            var h = 320f;
            var box = new Rect((UnityEngine.Screen.width - w) / 2f,
                (UnityEngine.Screen.height - h) / 2f, w, h);

            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var y = box.y + 22f;

            GUI.Label(new Rect(box.x, y, w, 40f), "Smart Traffic Intersection", UITheme.Title);
            y += 44f;
            GUI.Label(new Rect(box.x, y, w, 22f), "DQN vs Webster - live SUMO viewer",
                Centered(UITheme.Hint));
            y += 38f;

            var bx = box.x + 30f;
            var bw = w - 60f;

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Live visualization", UITheme.Button))
            {
                Go(Screen.Live);
            }
            y += rowH + gap;

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Run setup  (coming next)", UITheme.Button))
            {
                Go(Screen.Controls);
            }
            y += rowH + gap;

            if (GUI.Button(new Rect(bx, y, bw, rowH), "About this build", UITheme.Button))
            {
                Go(Screen.About);
            }
            y += rowH + gap + 6f;

            GUI.Label(new Rect(bx, y, bw, 20f), "hub: " + HubUrl, Centered(UITheme.Hint));
        }

        private void DrawControls()
        {
            var box = Sheet(560f, 260f, "Run setup");
            var y = box.y + 74f;
            GUI.Label(new Rect(box.x + 24f, y, box.width - 48f, 130f),
                "Not built yet. This screen will start an episode from inside Unity - controller, " +
                "scenario, seed, episode length and playback speed - by POSTing to /sessions on " +
                "the hub, the same endpoint the React dashboard uses.\n\n" +
                "Until then, start episodes from the dashboard at localhost:5173 or with curl; " +
                "this viewer picks up whatever session is running.",
                Wrapped(UITheme.Label));
            BackButton(box);
        }

        private void DrawAbout()
        {
            var box = Sheet(560f, 300f, "About");
            var y = box.y + 74f;
            GUI.Label(new Rect(box.x + 24f, y, box.width - 48f, 170f),
                "Renders a live SUMO episode from the hub's ws/unity channel at 1 Hz, with " +
                "client-side interpolation between frames.\n\n" +
                "The intersection, signals and scenery are generated at runtime from " +
                "config/network/intersection.*, so what you see cannot drift from the network " +
                "being simulated.\n\n" +
                "Each approach carries three signal heads because left, through and right are " +
                "independently controlled - those twelve movements are the agent's action space.",
                Wrapped(UITheme.Label));
            BackButton(box);
        }

        /// <summary>The thin bar shown while the live view is up.</summary>
        private void DrawLiveBar()
        {
            var w = UnityEngine.Screen.width;
            UITheme.Backdrop(new Rect(0f, 0f, w, 30f));
            if (GUI.Button(new Rect(w - 108f, 4f, 100f, 22f), "Menu  Esc", UITheme.Button))
            {
                Go(Screen.Menu);
            }
        }

        private Rect Sheet(float w, float h, string title)
        {
            var box = new Rect((UnityEngine.Screen.width - w) / 2f,
                (UnityEngine.Screen.height - h) / 2f, w, h);
            GUI.Box(box, GUIContent.none, UITheme.Panel);
            GUI.Label(new Rect(box.x, box.y + 20f, w, 34f), title, UITheme.Title);
            return box;
        }

        private void BackButton(Rect box)
        {
            if (GUI.Button(new Rect(box.x + 24f, box.yMax - 48f, 130f, 32f), "Back  Esc", UITheme.Button))
            {
                Go(Screen.Menu);
            }
        }

        private static GUIStyle Centered(GUIStyle s) =>
            new GUIStyle(s) { alignment = TextAnchor.MiddleCenter };

        private static GUIStyle Wrapped(GUIStyle s) =>
            new GUIStyle(s) { wordWrap = true, alignment = TextAnchor.UpperLeft };
    }
}
