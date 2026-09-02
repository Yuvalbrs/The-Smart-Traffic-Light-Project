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
        public enum Screen { Menu, Live, Dashboard, Train, Compare, Controls, About }

        public string HubUrl = "ws://127.0.0.1:8000/ws/unity";

        public Screen Current { get; private set; } = Screen.Menu;

        private TrafficViz _viz;
        private DashboardScreen _dash;
        private TrainScreen _train;
        private CompareScreen _compare;
        private HubApi _api;
        private EpisodeSummary _summary;
        // The previous poll's state. The end of a run is a TRANSITION, and polling for the
        // string "finished" would re-open the panel every second until the next episode starts.
        private string _prevSessionState = "unknown";
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
            // One API client for every screen: the model list the Train screen shows and the
            // scene list the dashboard offers must be the same lists, fetched once.
            _api = new HubApi(HubUrl);
            _api.RefreshControllers();
            _summary = new EpisodeSummary(_api);
        }

        private void Update()
        {
            // Session status drives the picker's readout; once a second is plenty for a 1 Hz feed.
            if (Time.unscaledTime >= _nextPoll)
            {
                _nextPoll = Time.unscaledTime + 1f;
                _session?.Refresh();
                // The live view has no session poll of its own; hand it the state so its HUD can
                // tell "episode finished" apart from "the feed died".
                if (_session != null)
                {
                    if (_viz != null) _viz.SessionState = _session.State;
                    if (_dash != null) _dash.SessionState = _session.State;

                    // The end of an episode, caught as an edge rather than a level. Only from a
                    // state that was actually RUNNING: on a cold start the hub reports the LAST
                    // session, and levelling on "finished" would greet the user with a summary of
                    // a run from yesterday before they had started anything.
                    var now = _session.State;
                    var ended = now == "finished" || now == "stopped" || now == "failed";
                    var wasRunning = _prevSessionState == "running" || _prevSessionState == "starting";

                    // Captured on ANY screen, not only Live. An episode that finishes while the
                    // user is reading the dashboard still produced results, and the summary is
                    // waiting for them when they come back to the picture. Drawing is gated to
                    // Live separately - capture and display are different decisions.
                    if (ended && wasRunning && _summary != null)
                    {
                        _summary.Show(_session.RunId, _session.RunningScenario,
                                      _session.RunningController, _session.Seed,
                                      _session.SimTime, now);
                    }

                    // A new run retires the old summary. Without this, starting the next episode
                    // leaves the previous run's numbers sitting over live traffic - the exact
                    // "correct data made to look like other data" defect this project has already
                    // paid for twice.
                    if (now == "running" || now == "starting") _summary?.Hide();

                    _prevSessionState = now;
                }
            }

            if (Current == Screen.Live && Input.GetKeyDown(KeyCode.Tab)) _overlay = !_overlay;

            // Draining the feed is main-thread work and must happen every frame the screen is up,
            // not inside OnGUI - OnGUI runs several times per frame for layout and events.
            _dash?.Tick();
            _train?.Tick();
            _summary?.Tick();

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
                _dash = new DashboardScreen(HubUrl) { Api = _api, Session = _session, Config = _cfg };
                _api.RefreshRuns();
            }
            if (screen == Screen.Train && _train == null) _train = new TrainScreen(_api);
            if (screen == Screen.Compare) _compare = new CompareScreen(_api);

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
                case Screen.Train: DrawTrain(); break;
                case Screen.Compare: DrawCompare(); break;
                case Screen.Live: DrawLiveBar(); break;
            }

            // Last, so it sits over the live HUD rather than under it - and only on Live, so it
            // can never cover the training curve or the comparison table.
            if (Current == Screen.Live)
            {
                _summary?.Draw(new Rect(0f, 0f, UnityEngine.Screen.width, UnityEngine.Screen.height));
            }
        }

        private void DrawMenu()
        {
            MenuBackdrop.Draw();
            // Every dimension here scales with the type in UITheme; the panel was sized around
            // 13 pt text and clipped its own buttons the moment the fonts grew.
            const float w = 560f, rowH = 58f, gap = 12f;
            var h = 626f;   // two more entries: Train and Compare
            var box = new Rect((UnityEngine.Screen.width - w) / 2f,
                (UnityEngine.Screen.height - h) / 2f, w, h);

            GUI.Box(box, GUIContent.none, UITheme.Panel);
            var y = box.y + 26f;

            GUI.Label(new Rect(box.x, y, w, 52f), "Smart Traffic Intersection", UITheme.Title);
            y += 58f;
            GUI.Label(new Rect(box.x, y, w, 26f), "DQN vs Webster - live SUMO viewer",
                Centered(UITheme.Hint));
            y += 44f;

            var bx = box.x + 36f;
            var bw = w - 72f;

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

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Train a controller", UITheme.Button))
            {
                Go(Screen.Train);
            }
            y += rowH + gap;

            if (GUI.Button(new Rect(bx, y, bw, rowH), "Compare controllers", UITheme.Button))
            {
                Go(Screen.Compare);
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
            MenuBackdrop.Draw();
            _dash?.Draw();
            NavBar();
        }

        private void DrawTrain()
        {
            MenuBackdrop.Draw();
            _train?.Draw(ScreenBox());
            NavBar();
        }

        private void DrawCompare()
        {
            MenuBackdrop.Draw();
            _compare?.Draw(ScreenBox());
            NavBar();
        }

        /// <summary>The area a full screen gets, leaving room for the nav bar underneath.</summary>
        private static Rect ScreenBox()
        {
            const float pad = 28f;
            var w = Mathf.Min(1560f, UnityEngine.Screen.width - pad * 2f);
            return new Rect((UnityEngine.Screen.width - w) / 2f, pad, w,
                UnityEngine.Screen.height - pad - 78f);
        }

        /// <summary>
        /// Live / Dashboard / Train / Compare, on every screen that has them.
        ///
        /// The React app puts these four in one tab strip and never makes you go back to a menu
        /// to change view; there is no reason this client should either, and during a demo the
        /// trip out to a menu is exactly where the thread of an explanation gets dropped.
        /// </summary>
        private void NavBar()
        {
            var y = UnityEngine.Screen.height - 64f;
            var labels = new[] { "3-D view", "Dashboard", "Train", "Compare" };
            var screens = new[] { Screen.Live, Screen.Dashboard, Screen.Train, Screen.Compare };
            const float bw = 190f, gap = 10f;
            var total = labels.Length * bw + (labels.Length - 1) * gap + 180f;
            var x = (UnityEngine.Screen.width - total) / 2f;

            for (var i = 0; i < labels.Length; i++)
            {
                if (GUI.Button(new Rect(x + i * (bw + gap), y, bw, 46f), labels[i],
                        Current == screens[i] ? UITheme.ButtonOn : UITheme.Button))
                {
                    Go(screens[i]);
                }
            }
            if (GUI.Button(new Rect(x + labels.Length * (bw + gap), y, 170f, 46f), "Menu",
                    UITheme.Button))
            {
                Go(Screen.Menu);
            }
        }

        private void DrawControls()
        {
            // Controls and About are full screens over an empty scene, same as the menu - without
            // the backdrop they sit on a flat void.
            MenuBackdrop.Draw();
            var w = RunSetupUI.PanelWidth;
            var h = RunSetupUI.PanelHeight;
            var box = new Rect((UnityEngine.Screen.width - w) / 2f,
                (UnityEngine.Screen.height - h) / 2f - 16f, w, h);

            if (RunSetupUI.Draw(box, ref _cfg, _session, showStop: true)) RunAndWatch();

            // 30f squeezed an 18 pt bold label with 11 px of vertical padding into a box that
            // could not hold it, so "Back" was clipped top and bottom. LineH asks the style.
            if (GUI.Button(new Rect(box.x, box.yMax + 8f, 130f, UITheme.LineH(UITheme.Button)),
                    "Back", UITheme.Button))
            {
                Go(Screen.Menu);
            }
        }

        private Vector2 _aboutScroll;

        private void DrawAbout()
        {
            MenuBackdrop.Draw();
            // Taller, and scrolled. At 560x300 with the larger type the last paragraph and the
            // whole keyboard list were simply unreachable - text that exists and cannot be read.
            var box = Sheet(820f, 560f, "About");
            const string body =
                "Renders a live SUMO episode from the hub's ws/unity channel at 1 Hz, with " +
                "client-side interpolation between frames.\n\n" +
                "The intersection, signals and scenery are generated at runtime from " +
                "config/network/intersection.*, so what you see cannot drift from the network " +
                "being simulated.\n\n" +
                "Each approach carries three signal heads because left, through and right are " +
                "independently controlled - those twelve movements are the agent's action space." +
                "\n\n" +
                "Keyboard\n" +
                "  D      switch between the 3-D view and the dashboard\n" +
                "  Tab    open the scene picker over the 3-D view\n" +
                "  Esc    close the picker, then leave the screen\n" +
                "  F      free camera; Esc returns to orbit\n" +
                "  1-6    camera presets\n" +
                "  wheel  zoom, right-drag orbit, middle-drag pan";

            var view = new Rect(box.x + 24f, box.y + 78f, box.width - 48f, box.height - 150f);
            var style = Wrapped(UITheme.Label);
            var height = style.CalcHeight(new GUIContent(body), view.width - 24f);
            _aboutScroll = GUI.BeginScrollView(view, _aboutScroll,
                new Rect(0f, 0f, view.width - 24f, height));
            GUI.Label(new Rect(0f, 0f, view.width - 24f, height), body, style);
            GUI.EndScrollView();
            BackButton(box);
        }

        /// <summary>Height of the live-view top bar. TrafficViz lays its HUD out below this.</summary>
        public const float TopBarHeight = 62f;

        /// <summary>The thin bar shown while the live view is up.</summary>
        private void DrawLiveBar()
        {
            var w = UnityEngine.Screen.width;
            UITheme.Backdrop(new Rect(0f, 0f, w, TopBarHeight));
            // 26f clipped the descender of "dqn-plain" - Heading is 20 pt bold and does not fit.
            GUI.Label(new Rect(20f, 18f, 620f, UITheme.LineH(UITheme.Heading)),
                "SMART TRAFFIC   -   " + _cfg.Scenario + "  /  " + _cfg.Controller, UITheme.Heading);

            // The 3-D view had no way out to the numbers except the D key, which nobody discovers
            // during a demo. Buttons laid out from the right edge so they cannot overlap the
            // title on a narrow window.
            //
            // Labels carry no key hints. "Dashboard  D" was read as a typo rather than as a
            // shortcut, which is a fair reading - a shortcut nobody asked about is noise on the
            // button. The keys still work; they are listed on the About screen instead.
            const float bh = 44f, by = 8f, gap = 10f;
            var x = w - 16f;

            x -= 140f;
            if (GUI.Button(new Rect(x, by, 140f, bh), "Menu", UITheme.Button))
            {
                Go(Screen.Menu);
            }

            x -= 170f + gap;
            if (GUI.Button(new Rect(x, by, 170f, bh), "Dashboard", UITheme.Button))
            {
                Go(Screen.Dashboard);
            }

            x -= 200f + gap;
            if (GUI.Button(new Rect(x, by, 200f, bh),
                    _overlay ? "Close" : "Change scene",
                    _overlay ? UITheme.ButtonOn : UITheme.Button))
            {
                _overlay = !_overlay;
            }

            // Stop AND clear, as one action. Stopping alone leaves the junction full of the
            // previous run's traffic, which is the state that makes "did it finish or did it
            // break?" unanswerable at a glance.
            x -= 130f + gap;
            if (GUI.Button(new Rect(x, by, 130f, bh), "Reset", UITheme.Button))
            {
                _session?.Stop();
                _viz?.ClearVehicles();
                _summary?.Hide();
                _prevSessionState = "stopped";   // a reset is not an episode ending; no summary
            }

            x -= 150f + gap;
            if (GUI.Button(new Rect(x, by, 150f, bh),
                    DayNight.IsNight ? "Day" : "Night", UITheme.Button))
            {
                DayNight.Toggle();
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
            if (GUI.Button(new Rect(box.x + 24f, box.yMax - 48f, 130f, UITheme.LineH(UITheme.Button)),
                    "Back", UITheme.Button))
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
