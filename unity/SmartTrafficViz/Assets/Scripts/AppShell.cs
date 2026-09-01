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
        public enum Screen { Menu, Live, Dashboard, Controls, About }

        public string HubUrl = "ws://127.0.0.1:8000/ws/unity";

        public Screen Current { get; private set; } = Screen.Menu;

        private TrafficViz _viz;
        private DashboardScreen _dash;
        private Camera _menuCam;
        private SessionControl _session;
        private RunConfig _cfg = RunConfig.Default;
        private bool _overlay;      // run-setup panel open on top of the live view
        private float _nextPoll;

        private void Start()
        {
            EnsureMenuCamera();
            _session = new SessionControl(HubUrl);
            _session.Refresh();
        }

        private void Update()
        {
            // Session status drives the picker's readout; once a second is plenty for a 1 Hz feed.
            if (Time.unscaledTime >= _nextPoll)
            {
                _nextPoll = Time.unscaledTime + 1f;
                _session?.Refresh();
            }

            if (Current == Screen.Live && Input.GetKeyDown(KeyCode.Tab)) _overlay = !_overlay;

            // Draining the feed is main-thread work and must happen every frame the screen is up,
            // not inside OnGUI - OnGUI runs several times per frame for layout and events.
            _dash?.Tick();

            // D toggles straight between the picture and the numbers, so a demo can move between
            // them without going back out to the menu each time.
            if (Input.GetKeyDown(KeyCode.D) && (Current == Screen.Live || Current == Screen.Dashboard))
            {
                Go(Current == Screen.Live ? Screen.Dashboard : Screen.Live);
            }

            // Escape backs out one level: first close the overlay, then leave the screen.
            if (Input.GetKeyDown(KeyCode.Escape) && !InFreeCamera())
            {
                if (_overlay) _overlay = false;
                else if (Current != Screen.Menu) Go(Screen.Menu);
            }
        }

        /// <summary>Starts (or switches to) the chosen scene + controller and shows it.</summary>
        private void RunAndWatch()
        {
            _session.Switch(_cfg);
            _overlay = false;
            if (Current != Screen.Live) Go(Screen.Live);
        }

        private void OnDestroy()
        {
            _dash?.Dispose();
            _dash = null;
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
            if (Current == Screen.Dashboard && _dash != null)
            {
                // Leaving the screen releases the socket, exactly as leaving Live releases the
                // 3-D view's - an idle screen must not keep a subscription open on the hub.
                _dash.Dispose();
                _dash = null;
            }

            Current = screen;

            if (screen == Screen.Dashboard)
            {
                _dash = new DashboardScreen(HubUrl);
            }

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
                case Screen.Dashboard: DrawDashboard(); break;
                case Screen.Live: DrawLiveBar(); break;
            }
        }

        private void DrawMenu()
        {
            const float w = 420f, rowH = 44f, gap = 10f;
            var h = 374f;   // one row taller since the dashboard entry joined the menu
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

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Live dashboard", UITheme.Button))
            {
                Go(Screen.Dashboard);
            }
            y += rowH + gap;

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Choose scene + controller", UITheme.Button))
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

        private void DrawDashboard()
        {
            _dash?.Draw();
            if (GUI.Button(new Rect(20f, UnityEngine.Screen.height - 44f, 150f, 30f),
                    "Back  Esc", UITheme.Button))
            {
                Go(Screen.Menu);
            }
            if (GUI.Button(new Rect(180f, UnityEngine.Screen.height - 44f, 170f, 30f),
                    "3-D view  D", UITheme.Button))
            {
                Go(Screen.Live);
            }
        }

        private void DrawControls()
        {
            var w = RunSetupUI.PanelWidth;
            var h = RunSetupUI.PanelHeight;
            var box = new Rect((UnityEngine.Screen.width - w) / 2f,
                (UnityEngine.Screen.height - h) / 2f - 16f, w, h);

            if (RunSetupUI.Draw(box, ref _cfg, _session, showStop: true)) RunAndWatch();

            if (GUI.Button(new Rect(box.x, box.yMax + 8f, 130f, 30f), "Back  Esc", UITheme.Button))
            {
                Go(Screen.Menu);
            }
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

        /// <summary>Height of the live-view top bar. TrafficViz lays its HUD out below this.</summary>
        public const float TopBarHeight = 34f;

        /// <summary>The thin bar shown while the live view is up.</summary>
        private void DrawLiveBar()
        {
            var w = UnityEngine.Screen.width;
            UITheme.Backdrop(new Rect(0f, 0f, w, TopBarHeight));
            GUI.Label(new Rect(14f, 7f, 460f, 20f),
                "SMART TRAFFIC   -   " + _cfg.Scenario + "  /  " + _cfg.Controller, UITheme.Heading);

            if (GUI.Button(new Rect(w - 250f, 4f, 128f, 26f),
                    _overlay ? "Close  Tab" : "Change scene  Tab",
                    _overlay ? UITheme.ButtonOn : UITheme.Button))
            {
                _overlay = !_overlay;
            }
            if (GUI.Button(new Rect(w - 116f, 4f, 104f, 26f), "Menu  Esc", UITheme.Button))
            {
                Go(Screen.Menu);
            }

            if (!_overlay) return;

            // Switching without leaving the 3-D view is the point: Webster then DQN on the same
            // scene and seed, back to back, is the comparison the whole project rests on.
            var pw = RunSetupUI.PanelWidth;
            var ph = RunSetupUI.PanelHeight;
            var box = new Rect(w - pw - 16f, TopBarHeight + 10f, pw, ph);
            if (RunSetupUI.Draw(box, ref _cfg, _session, showStop: true)) RunAndWatch();
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
