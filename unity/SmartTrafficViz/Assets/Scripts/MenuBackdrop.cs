// A quiet star field behind the menu screens.
//
// The menu used to sit on a flat near-black clear colour, which reads as "nothing has loaded yet"
// rather than as a background. This gives it depth without competing with the panel in front of
// it: a deep vertical gradient, two very faint nebula washes, and a field of small stars.
//
// Deliberately understated. Everything here is low-alpha and low-contrast, the twinkle is a few
// percent of brightness rather than a blink, and there is no motion across the screen - a
// background that draws the eye is a background that pulls attention off the thing being
// demonstrated.
//
// Drawn with IMGUI textures rather than 3-D geometry or a skybox, for the same reason the rest of
// this UI is IMGUI: it is a handful of draw calls of generated pixels, it cannot fail the way a
// missing shader can (see IntersectionScene.ResolveShader for why that matters in a player
// build), and it needs no scene, camera or asset to exist.

using UnityEngine;

namespace SmartTraffic
{
    public static class MenuBackdrop
    {
        //: Fixed, so the sky is identical in every screenshot and every run.
        private const int Seed = 20260902;
        private const int StarCount = 190;

        private static Texture2D _gradient, _glow, _dot;
        private static Star[] _stars;

        private struct Star
        {
            public float X, Y;       // normalised 0..1, so the field rescales with the window
            public float Size;
            public float Alpha;
            public float Phase;      // twinkle offset, so they do not pulse in unison
        }

        private static void Ensure()
        {
            // Same staleness guard the rest of the UI uses: these statics outlive Play mode but
            // the textures they point at do not.
            if (_gradient != null && _stars != null && _dot != null && _glow != null) return;

            _gradient = BuildGradient();
            _glow = BuildGlow();
            _dot = Solid(Color.white);
            _stars = BuildStars();
        }

        private static Texture2D Solid(Color c)
        {
            var t = new Texture2D(1, 1) { hideFlags = HideFlags.HideAndDontSave };
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        /// <summary>A 1 x N vertical ramp, stretched across the screen by the sampler.</summary>
        private static Texture2D BuildGradient()
        {
            const int h = 128;
            var t = new Texture2D(1, h, TextureFormat.RGBA32, false)
            {
                hideFlags = HideFlags.HideAndDontSave,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };

            // Texture row 0 is the BOTTOM of the image; GUI space runs top-down. Building the ramp
            // bottom-up here keeps the darker end at the top of the screen where the title sits.
            var top = new Color(0.024f, 0.031f, 0.055f);      // near-black indigo
            var mid = new Color(0.043f, 0.055f, 0.094f);
            var bottom = new Color(0.075f, 0.082f, 0.125f);   // a touch warmer, like distant haze

            for (var y = 0; y < h; y++)
            {
                var v = y / (float)(h - 1);                    // 0 = bottom of screen
                var c = v < 0.5f
                    ? Color.Lerp(bottom, mid, v * 2f)
                    : Color.Lerp(mid, top, (v - 0.5f) * 2f);
                t.SetPixel(0, y, c);
            }
            t.Apply();
            return t;
        }

        /// <summary>A soft radial falloff used for the nebula washes, white with an alpha ramp.</summary>
        private static Texture2D BuildGlow()
        {
            const int n = 96;
            var t = new Texture2D(n, n, TextureFormat.RGBA32, false)
            {
                hideFlags = HideFlags.HideAndDontSave,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            var c = (n - 1) / 2f;
            for (var y = 0; y < n; y++)
            {
                for (var x = 0; x < n; x++)
                {
                    var d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c)) / c;
                    // Squared falloff: a linear one has a visible edge where it reaches zero.
                    var a = Mathf.Clamp01(1f - d);
                    t.SetPixel(x, y, new Color(1f, 1f, 1f, a * a));
                }
            }
            t.Apply();
            return t;
        }

        private static Star[] BuildStars()
        {
            var rng = new System.Random(Seed);
            var stars = new Star[StarCount];
            for (var i = 0; i < StarCount; i++)
            {
                // Most stars are faint; a handful are brighter. A uniform brightness reads as
                // noise rather than as a sky.
                var r = (float)rng.NextDouble();
                var bright = r > 0.93f;
                stars[i] = new Star
                {
                    X = (float)rng.NextDouble(),
                    Y = (float)rng.NextDouble(),
                    Size = bright ? 2f : 1f,
                    Alpha = bright
                        ? 0.55f + (float)rng.NextDouble() * 0.35f
                        : 0.12f + (float)rng.NextDouble() * 0.30f,
                    Phase = (float)rng.NextDouble() * Mathf.PI * 2f,
                };
            }
            return stars;
        }

        /// <summary>Fills the whole screen. Call first, so everything else draws on top.</summary>
        public static void Draw()
        {
            Ensure();

            var w = UnityEngine.Screen.width;
            var h = UnityEngine.Screen.height;
            var full = new Rect(0f, 0f, w, h);
            var prev = GUI.color;

            GUI.color = Color.white;
            GUI.DrawTexture(full, _gradient);

            // Two washes, cool and warm, well apart and both barely there.
            DrawGlow(w * 0.22f, h * 0.30f, w * 0.62f, new Color(0.28f, 0.42f, 0.85f, 0.050f));
            DrawGlow(w * 0.84f, h * 0.74f, w * 0.52f, new Color(0.55f, 0.32f, 0.70f, 0.038f));

            var t = Time.unscaledTime;
            foreach (var s in _stars)
            {
                // +-8% brightness. Enough to stop the field looking like a printed texture,
                // far short of anything that blinks.
                var a = s.Alpha * (0.92f + 0.08f * Mathf.Sin(t * 0.7f + s.Phase));
                GUI.color = new Color(0.86f, 0.91f, 1f, a);
                GUI.DrawTexture(new Rect(s.X * w, s.Y * h, s.Size, s.Size), _dot);
            }

            GUI.color = prev;
        }

        private static void DrawGlow(float cx, float cy, float size, Color tint)
        {
            GUI.color = tint;
            GUI.DrawTexture(new Rect(cx - size / 2f, cy - size / 2f, size, size), _glow);
        }
    }
}
