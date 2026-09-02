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

        // Midday: a high, slightly warm sun over a blue sky.
        private static readonly Color DaySun = new Color(1.00f, 0.97f, 0.90f);
        private static readonly Color DayAmbient = new Color(0.42f, 0.45f, 0.50f);
        private static readonly Color DaySky = new Color(0.42f, 0.58f, 0.75f);

        // Night: a dim blue "moon" from the opposite side. Not black - a scene with no fill light
        // reads as a bug, and the audience still has to be able to see the road.
        private static readonly Color NightSun = new Color(0.42f, 0.52f, 0.78f);
        private static readonly Color NightAmbient = new Color(0.13f, 0.15f, 0.22f);
        private static readonly Color NightSky = new Color(0.045f, 0.055f, 0.095f);

        /// <summary>Forget the previous scene's lights. Call before building a new one.</summary>
        public static void Reset()
        {
            _sun = null;
            Lamps.Clear();
            LampGlass.Clear();
        }

        public static void RegisterSun(Light sun)
        {
            _sun = sun;
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

            if (_sun != null)
            {
                _sun.color = IsNight ? NightSun : DaySun;
                _sun.intensity = IsNight ? 0.32f : 1.05f;
                // Low from the opposite quarter at night, so the long shadows read as moonlight
                // rather than as the sun having simply dimmed in place.
                _sun.transform.rotation = IsNight
                    ? Quaternion.Euler(24f, 205f, 0f)
                    : Quaternion.Euler(52f, 35f, 0f);
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
