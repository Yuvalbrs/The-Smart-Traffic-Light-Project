// T-05-03 - a small IMGUI theme.
//
// The client's UI is OnGUI rather than uGUI canvases on purpose: every screen here is a handful
// of buttons and labels over a 3-D view, and IMGUI keeps that as reviewable text instead of
// scene-serialised prefabs that cannot be diffed. What IMGUI does not give you is a look, so the
// styles live here rather than being rebuilt inline in three different components.
//
// Styles are created lazily inside OnGUI: GUIStyle construction needs the GUI system live, so
// building them in Awake is not safe.

using UnityEngine;

namespace SmartTraffic
{
    public static class UITheme
    {
        public static readonly Color Accent = new Color(0.30f, 0.65f, 1.00f);
        public static readonly Color Ink = new Color(0.92f, 0.94f, 0.97f);
        public static readonly Color Dim = new Color(0.62f, 0.66f, 0.72f);

        private static GUIStyle _button, _buttonOn, _panel, _title, _label, _hint, _heading;
        private static Texture2D _panelTex, _buttonTex, _hoverTex, _onTex;

        private static Texture2D Solid(Color c)
        {
            var t = new Texture2D(1, 1) { hideFlags = HideFlags.HideAndDontSave };
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        /// <summary>Rebuilds styles if they are missing or were destroyed on Play-mode exit.</summary>
        private static void Ensure()
        {
            if (_button != null && _panelTex != null) return;

            _panelTex = Solid(new Color(0.07f, 0.09f, 0.12f, 0.88f));
            _buttonTex = Solid(new Color(0.16f, 0.19f, 0.24f, 0.95f));
            _hoverTex = Solid(new Color(0.24f, 0.29f, 0.36f, 0.98f));
            _onTex = Solid(new Color(0.13f, 0.38f, 0.62f, 0.98f));

            _button = new GUIStyle(GUI.skin.button)
            {
                fontSize = 13,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                padding = new RectOffset(10, 10, 7, 7),
                border = new RectOffset(2, 2, 2, 2),
            };
            _button.normal.background = _buttonTex;
            _button.normal.textColor = Ink;
            _button.hover.background = _hoverTex;
            _button.hover.textColor = Color.white;
            _button.active.background = _onTex;
            _button.active.textColor = Color.white;

            _buttonOn = new GUIStyle(_button);
            _buttonOn.normal.background = _onTex;
            _buttonOn.normal.textColor = Color.white;
            _buttonOn.hover.background = _onTex;

            _panel = new GUIStyle { padding = new RectOffset(16, 16, 14, 14) };
            _panel.normal.background = _panelTex;

            _title = new GUIStyle(GUI.skin.label)
            {
                fontSize = 30,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
            };
            _title.normal.textColor = Color.white;

            _heading = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold };
            _heading.normal.textColor = Accent;

            _label = new GUIStyle(GUI.skin.label) { fontSize = 13 };
            _label.normal.textColor = Ink;

            _hint = new GUIStyle(GUI.skin.label) { fontSize = 12 };
            _hint.normal.textColor = Dim;
        }

        public static GUIStyle Button { get { Ensure(); return _button; } }
        public static GUIStyle ButtonOn { get { Ensure(); return _buttonOn; } }
        public static GUIStyle Panel { get { Ensure(); return _panel; } }
        public static GUIStyle Title { get { Ensure(); return _title; } }
        public static GUIStyle Heading { get { Ensure(); return _heading; } }
        public static GUIStyle Label { get { Ensure(); return _label; } }
        public static GUIStyle Hint { get { Ensure(); return _hint; } }

        /// <summary>Fills a rect with the panel background - for bars behind controls.</summary>
        public static void Backdrop(Rect rect)
        {
            Ensure();
            GUI.DrawTexture(rect, _panelTex);
        }
    }
}
