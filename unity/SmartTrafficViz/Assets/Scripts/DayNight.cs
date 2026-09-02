// Day and night for the 3-D view.
//
// One switch changes four things at once: the sun's colour, angle and strength; the ambient
// light every unlit face is bathed in; the sky the camera clears to; and whether the street
// lamps are burning. Changing only the sun gives you a dark scene rather than a night scene -
// the giveaway is that shadowed faces stay bright, because ambient light is what fills them.
//
// The scene is generated at runtime and torn down when the viewer leaves the Live screen, so
// this cannot hold references across a rebuild. Everything registers itself as it is built and
// Reset() is called before each rebuild; a stale Light from a destroyed scene is null, and Unity's
// fake-null makes that safe to test but not safe to keep.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public static class DayNight
    {
        public static bool IsNight { get; private set; }

        private static Light _sun;
        private static readonly List<Light> Lamps = new List<Light>();
        private static readonly List<Renderer> LampGlass = new List<Renderer>();
        private static Material _lampOn, _lampOff;

        // The visible disc in the sky. The directional light on its own lights the scene from
        // nowhere: every face is correctly lit and the sky it is lit from is empty, which reads as
        // a missing object rather than as midday.
        private static Transform _body;
        private static Renderer _bodyRenderer;
        private static Material _sunMat, _moonMat;

        /// <summary>How far out the disc sits. Beyond the far ridge ring - which reaches 940 m,
        /// so no peak can stand in front of it - and inside the camera's 2600 m far clip, which
        /// was widened from 1500 m for exactly this reason (see TrafficViz.EnsureCamera).</summary>
        private const float BodyDistance = 1250f;
        private const float SunRadius = 55f;
        private const float MoonRadius = 44f;

        private static readonly Color SunDisc = new Color(1.00f, 0.93f, 0.72f);
        private static readonly Color MoonDisc = new Color(0.86f, 0.89f, 0.97f);

        // Midday: a high, slightly warm sun over a blue sky.
        //
        // Ambient is the number that decides whether the scene reads as noon or as overcast dusk.
        // It fills every face the key light does not reach, so raising ONLY the sun leaves the
        // shadow side exactly as dark and just blows out the lit side. This was 0.42/0.45/0.50 -
        // a dim sky-bounce - and the junction read as permanently shadowed even at midday.
        private static readonly Color DaySun = new Color(1.00f, 0.97f, 0.91f);
        private static readonly Color DayAmbient = new Color(0.64f, 0.67f, 0.72f);
        private static readonly Color DaySky = new Color(0.52f, 0.69f, 0.87f);

        // Night: a dim blue "moon" from the opposite side. Not black - a scene with no fill light
        // reads as a bug, and the audience still has to be able to see the road.
        private static readonly Color NightSun = new Color(0.42f, 0.52f, 0.78f);
        private static readonly Color NightAmbient = new Color(0.13f, 0.15f, 0.22f);
        private static readonly Color NightSky = new Color(0.045f, 0.055f, 0.095f);

        /// <summary>Forget the previous scene's lights. Call before building a new one.</summary>
        public static void Reset()
        {
            _sun = null;
            _body = null;
            _bodyRenderer = null;
            Lamps.Clear();
            LampGlass.Clear();
        }

        public static void RegisterSun(Light sun)
        {
            _sun = sun;
            Apply();
        }

        /// <summary>The sphere that stands in for the sun by day and the moon by night.</summary>
        public static void RegisterBody(Transform body, Renderer renderer)
        {
            _body = body;
            _bodyRenderer = renderer;
            Apply();
        }

        /// <summary>A street lamp: its point light, and the lens that should glow when lit.</summary>
        public static void RegisterLamp(Light lamp, Renderer glass)
        {
            if (lamp != null) Lamps.Add(lamp);
            if (glass != null) LampGlass.Add(glass);
            Apply();
        }

        public static void Toggle()
        {
            IsNight = !IsNight;
            Apply();
        }

        public static void Apply()
        {
            RenderSettings.ambientLight = IsNight ? NightAmbient : DayAmbient;

            // ONE rotation drives both the light and the disc, so the sun can never be drawn in a
            // quarter of the sky the shadows say it is not in. Low from the opposite quarter at
            // night, so the long shadows read as moonlight rather than as the sun having simply
            // dimmed in place - but 30 deg, not the 24 deg this used before the disc existed: the
            // far ridge ring tops out at ~24 deg of elevation, and a moon on that vector sat in
            // the mountains instead of above them.
            var rotation = IsNight
                ? Quaternion.Euler(30f, 205f, 0f)
                : Quaternion.Euler(52f, 35f, 0f);

            if (_sun != null)
            {
                _sun.color = IsNight ? NightSun : DaySun;
                _sun.intensity = IsNight ? 0.32f : 1.30f;
                _sun.transform.rotation = rotation;

                // A fully opaque cast shadow is the other half of "too shadowy": at noon the sky
                // is a huge second source, so real daylight shadows are filled in, not black.
                // Softening the shadow keeps the shape - which is what makes the 3-D read - while
                // letting the road surface under a signal head stay legible.
                _sun.shadowStrength = IsNight ? 0.80f : 0.50f;
            }

            if (_body != null)
            {
                // The disc belongs where the light comes FROM. A directional light travels along
                // its own forward vector, so the source is at -forward.
                _body.position = -(rotation * Vector3.forward) * BodyDistance;
                var d = (IsNight ? MoonRadius : SunRadius) * 2f;
                _body.localScale = new Vector3(d, d, d);
            }

            EnsureBodyMaterials();
            if (_bodyRenderer != null)
            {
                _bodyRenderer.sharedMaterial = IsNight ? _moonMat : _sunMat;
            }

            var cam = Camera.main;
            if (cam != null)
            {
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = IsNight ? NightSky : DaySky;
            }

            foreach (var lamp in Lamps)
            {
                if (lamp != null) lamp.enabled = IsNight;
            }

            EnsureLampMaterials();
            foreach (var glass in LampGlass)
            {
                if (glass != null) glass.sharedMaterial = IsNight ? _lampOn : _lampOff;
            }
        }

        private static void EnsureBodyMaterials()
        {
            // Same lifetime problem as the lamp materials: these statics outlive Play mode but the
            // Materials they point at do not.
            if (_sunMat != null && _moonMat != null) return;
            var shader = IntersectionScene.LitShader;
            if (shader == null) return;

            // Emissive, because the disc is a light source: lit normally it would be shaded by the
            // very light it represents, and the half facing the scene would be dark.
            _sunMat = new Material(shader) { color = SunDisc };
            _sunMat.EnableKeyword("_EMISSION");
            _sunMat.SetColor("_EmissionColor", SunDisc * 3.0f);

            _moonMat = new Material(shader) { color = MoonDisc };
            _moonMat.EnableKeyword("_EMISSION");
            _moonMat.SetColor("_EmissionColor", MoonDisc * 1.6f);
        }

        private static void EnsureLampMaterials()
        {
            // Rebuilt when the shader or a material has been destroyed - these statics outlive
            // Play mode but the objects they point at do not.
            if (_lampOn != null && _lampOff != null) return;
            var shader = IntersectionScene.LitShader;
            if (shader == null) return;

            var warm = new Color(1.00f, 0.90f, 0.68f);
            _lampOn = new Material(shader) { color = warm };
            _lampOn.EnableKeyword("_EMISSION");
            _lampOn.SetColor("_EmissionColor", warm * 2.4f);

            _lampOff = new Material(shader) { color = new Color(0.55f, 0.55f, 0.52f) };
        }
    }
}
