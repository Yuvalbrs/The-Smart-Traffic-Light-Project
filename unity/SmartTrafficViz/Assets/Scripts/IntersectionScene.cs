// T-05-03 - the intersection, built procedurally at runtime.
//
// Nothing here is authored in the Editor: the geometry is derived from config/network/
// intersection.nod.xml + intersection.edg.xml, so the rendered scene cannot drift away from the
// network SUMO is actually simulating. Open the project, press Play.
//
// Coordinate mapping. SUMO is 2-D with x east and y north; Unity is y-up, so SUMO (x, y) becomes
// Unity (x, 0, y). SUMO headings are degrees clockwise from north, which is exactly Unity's
// y-Euler convention, so angles carry across unchanged.
//
// Lane placement follows two SUMO rules, checked against a recorded frame rather than assumed:
//   * lane index 0 is the RIGHTMOST in the direction of travel (intersection.edg.xml);
//   * lanes sit to the right of the travel direction, so each approach occupies one half of its
//     arm and the opposing exit occupies the other.
// For the north approach (heading south) that puts lane 2 - the leftmost, left-turn-only lane -
// nearest the centreline at x = -1.6, which is where vehicle v120 on lane n_t_2 really was.
//
// Each approach carries ONE signal on a mast arm with THREE lamps, one per movement. That is not
// decoration: left / through / right are independently signalled (a frame really does show
// M0 red, M1 red, M2 green on the same approach) and those twelve movements are the agent's
// action space. Collapsing them to one lamp would hide what the controller actually does.

using System.Collections.Generic;
using UnityEngine;

namespace SmartTraffic
{
    public static class IntersectionScene
    {
        public const float ArmLength = 150f;    // intersection.nod.xml: N/E/S/W are 150 m out
        public const int LanesPerEdge = 3;      // intersection.edg.xml numLanes="3"
        public const float LaneWidth = 3.2f;    // SUMO default
        public const float HalfRoad = LanesPerEdge * LaneWidth;        // 9.6 m per direction
        public const float RoadWidth = HalfRoad * 2f;                  // both directions: 19.2 m
        public const float StopLine = HalfRoad + 1.2f;                 // just outside the junction

        /// <summary>Lane centre offsets from the centreline, index 0 = rightmost = furthest out.</summary>
        private static readonly float[] LaneOffsets = { 8.0f, 4.8f, 1.6f };

        /// <summary>M0..M11 in the canonical N,E,S,W x (left, through, right) order.</summary>
        public static readonly string[] MovementIds =
        {
            "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11",
        };

        public static readonly Color Green = new Color(0.15f, 0.95f, 0.35f);
        public static readonly Color Yellow = new Color(1.00f, 0.85f, 0.10f);
        public static readonly Color Red = new Color(1.00f, 0.20f, 0.18f);

        private static readonly Color Asphalt = new Color(0.20f, 0.21f, 0.23f);
        private static readonly Color Ground = new Color(0.16f, 0.22f, 0.16f);
        private static readonly Color MarkingPaint = new Color(0.92f, 0.92f, 0.88f);
        private static readonly Color Metal = new Color(0.22f, 0.23f, 0.25f);

        private static Shader _shader;

        /// <summary>Name of the material asset in Resources/ that pins a shader into the build.</summary>
        internal const string PinnedMaterial = "ScenePinMaterial";

        private static bool _shaderResolved;

        internal static Shader LitShader
        {
            get
            {
                if (_shaderResolved) return _shader;
                _shaderResolved = true;
                return _shader = ResolveShader();
            }
        }

        /// <summary>Find a usable shader, or null if the build genuinely has none.
        ///
        /// Resources FIRST, and that ordering is the whole point. `Shader.Find` alone returns null
        /// in a player build for any shader no asset references - the Editor has every built-in
        /// shader loaded, a build only ships what something points at. That is not hypothetical
        /// here: the first standalone build threw `ArgumentNullException: Parameter name: shader`
        /// on the very first Paint call, so BuildStatic never finished, the 3-D scene was never
        /// created, and Update then NREd on the null socket every frame - 185 MB of log in an hour.
        /// A material asset under Resources/ is what actually pins the shader in; the Find calls
        /// below are the fallback, not the mechanism.</summary>
        private static Shader ResolveShader()
        {
            var pinned = Resources.Load<Material>(PinnedMaterial);
            if (pinned != null && pinned.shader != null) return pinned.shader;

            foreach (var name in new[]
                     {
                         "Universal Render Pipeline/Lit", "Standard",
                         "Legacy Shaders/Diffuse", "Sprites/Default",
                     })
            {
                var found = Shader.Find(name);
                if (found != null) return found;
            }

            Debug.LogError(
                "[scene] no usable shader in this build - geometry will render with default " +
                "materials. Expected Resources/" + PinnedMaterial + " to pin one.");
            return null;
        }

        public static Color ColorFor(string name)
        {
            switch (name)
            {
                case "green": return Green;
                case "yellow": return Yellow;
                default: return Red;
            }
        }

        /// <summary>Builds ground, roads, markings and signal masts. Returns the root object.</summary>
        public static GameObject BuildStatic()
        {
            var root = new GameObject("Intersection");

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(root.transform);
            ground.transform.localScale = new Vector3(ArmLength / 2.5f, 1f, ArmLength / 2.5f);
            ground.transform.position = new Vector3(0f, -0.06f, 0f);
            Paint(ground, Ground);

            Slab(root, "Junction", new Vector3(RoadWidth, 0.1f, RoadWidth), Vector3.zero, Asphalt);

            var armLen = ArmLength - HalfRoad;
            var armMid = HalfRoad + armLen / 2f;
            Slab(root, "Arm_N", new Vector3(RoadWidth, 0.1f, armLen), new Vector3(0f, 0f, armMid), Asphalt);
            Slab(root, "Arm_S", new Vector3(RoadWidth, 0.1f, armLen), new Vector3(0f, 0f, -armMid), Asphalt);
            Slab(root, "Arm_E", new Vector3(armLen, 0.1f, RoadWidth), new Vector3(armMid, 0f, 0f), Asphalt);
            Slab(root, "Arm_W", new Vector3(armLen, 0.1f, RoadWidth), new Vector3(-armMid, 0f, 0f), Asphalt);

            BuildMarkings(root);
            return root;
        }

        /// <summary>Centre lines, lane dashes and stop bars - the cues that make flow readable.</summary>
        private static void BuildMarkings(GameObject root)
        {
            var markings = new GameObject("Markings").transform;
            markings.SetParent(root.transform);

            for (var approach = 0; approach < 4; approach++)
            {
                var axis = AxisOf(approach);            // unit vector pointing out along the arm
                var side = new Vector3(-axis.z, 0f, axis.x); // 90 deg left of it, in-plane

                // Solid centre line: divides the approach from the opposing exit.
                Marking(markings, "Centre", axis * (HalfRoad + (ArmLength - HalfRoad) / 2f),
                    0.3f, ArmLength - HalfRoad, axis);

                // Dashed lane separators at +-3.2 and +-6.4 from the centreline, both halves.
                foreach (var offset in new[] { -6.4f, -3.2f, 3.2f, 6.4f })
                {
                    for (var d = HalfRoad + 4f; d < ArmLength - 4f; d += 12f)
                    {
                        var pos = axis * (d + 3f) + side * offset;
                        Marking(markings, "Dash", pos, 0.22f, 6f, axis);
                    }
                }

                // Stop bar across this approach's three lanes only (its half of the arm).
                var stopCentre = axis * StopLine + side * (-HalfRoad / 2f);
                Marking(markings, "StopBar", stopCentre, HalfRoad, 0.7f, axis);
            }
        }

        /// <summary>
        /// A real three-lamp signal head: red on top, amber in the middle, green at the bottom,
        /// with only the current aspect lit. Position carries the state as well as colour, which
        /// is how a driver reads one - and it stays legible when the colour is small on screen.
        /// </summary>
        public sealed class SignalHead
        {
            public Renderer RedLamp, AmberLamp, GreenLamp;
            private LampMaterials _mats;

            internal SignalHead(LampMaterials mats) { _mats = mats; }

            /// <summary>Lights the aspect named by the wire ("red" | "yellow" | "green").</summary>
            public void Set(string aspect)
            {
                RedLamp.sharedMaterial = aspect == "red" ? _mats.RedOn : _mats.RedOff;
                AmberLamp.sharedMaterial = aspect == "yellow" ? _mats.AmberOn : _mats.AmberOff;
                GreenLamp.sharedMaterial = aspect == "green" ? _mats.GreenOn : _mats.GreenOff;
            }
        }

        internal sealed class LampMaterials
        {
            public Material RedOn, RedOff, AmberOn, AmberOff, GreenOn, GreenOff;

            public LampMaterials()
            {
                RedOn = Lit(Red); RedOff = Dark(Red);
                AmberOn = Lit(Yellow); AmberOff = Dark(Yellow);
                GreenOn = Lit(Green); GreenOff = Dark(Green);
            }

            // Emissive, so a lit aspect still reads as "on" on the shadowed side of the housing
            // rather than depending on where the key light happens to fall.
            private static Material Lit(Color c)
            {
                var m = new Material(LitShader) { color = c };
                m.EnableKeyword("_EMISSION");
                m.SetColor("_EmissionColor", c * 2.2f);
                return m;
            }

            private static Material Dark(Color c)
            {
                return new Material(LitShader) { color = c * 0.16f };
            }
        }

        /// <summary>
        /// One mast arm per approach - pole at the kerb, beam over the three approach lanes, and
        /// a three-lamp head hanging above each lane. Returns the heads keyed by movement id.
        /// </summary>
        public static Dictionary<string, SignalHead> BuildSignalHeads(Transform parent)
        {
            var heads = new Dictionary<string, SignalHead>();
            var mats = new LampMaterials();
            var signals = new GameObject("Signals").transform;
            signals.SetParent(parent);

            const float poleH = 13f;
            const float beamH = 12f;
            const float headTop = 10.6f;   // top lamp height
            const float lampGap = 1.9f;    // vertical spacing between aspects

            for (var approach = 0; approach < 4; approach++)
            {
                var axis = AxisOf(approach);
                var side = new Vector3(-axis.z, 0f, axis.x);
                // The approach occupies the half to the RIGHT of its travel direction. Travel is
                // inward (-axis), so that half lies along +side, and the kerb is at +HalfRoad.
                var baseAt = axis * StopLine + side * (HalfRoad + 1.6f);

                Slab(signals.gameObject, "Pole_" + approach,
                    new Vector3(0.55f, poleH, 0.55f), baseAt + Vector3.up * (poleH / 2f), Metal)
                    .transform.SetParent(signals);

                Slab(signals.gameObject, "Beam_" + approach,
                    Size(axis, 0.4f, HalfRoad + 1.6f),
                    axis * StopLine + side * (HalfRoad / 2f) + Vector3.up * beamH, Metal)
                    .transform.SetParent(signals);

                for (var turn = 0; turn < 3; turn++)
                {
                    // turn 0 = left = leftmost lane = index 2 = nearest the centreline.
                    var laneOffset = LaneOffsets[2 - turn];
                    var id = MovementIds[approach * 3 + turn];
                    var at = axis * StopLine + side * laneOffset;

                    // Hanger from the beam down to the head.
                    Slab(signals.gameObject, "Hanger_" + id, new Vector3(0.22f, 1.4f, 0.22f),
                        at + Vector3.up * (beamH - 0.7f), Metal).transform.SetParent(signals);

                    // The housing. Deliberately oversized - a true 1.2 m head is a couple of
                    // pixels at the ~280 m camera range, and the aspect is the point of drawing it.
                    Slab(signals.gameObject, "Housing_" + id,
                        new Vector3(2.1f, 6.4f, 1.3f), at + Vector3.up * (headTop - lampGap),
                        new Color(0.12f, 0.13f, 0.14f)).transform.SetParent(signals);

                    var head = new SignalHead(mats)
                    {
                        RedLamp = Lamp(signals, "Red_" + id, at, headTop, axis),
                        AmberLamp = Lamp(signals, "Amber_" + id, at, headTop - lampGap, axis),
                        GreenLamp = Lamp(signals, "Green_" + id, at, headTop - 2f * lampGap, axis),
                    };
                    head.Set("red");
                    heads[id] = head;

                    BuildSign(signals, id, at + Vector3.up * (headTop + 3.6f), axis, turn);
                }
            }
            return heads;
        }

        /// <summary>
        /// The blue direction disc above a head, with an arrow pointing where that movement goes.
        ///
        /// Which way is "left" is not cosmetic and is easy to get backwards. The sign faces the
        /// oncoming driver, so its plane is spanned by world up and <c>Cross(up, axis)</c>. For
        /// the north approach that in-plane +X is east - and a driver heading south has east on
        /// their left. So local +X is the driver's LEFT on every approach, and the left-turn arrow
        /// rotates -90 degrees from vertical, the right-turn arrow +90.
        /// </summary>
        private static void BuildSign(Transform parent, string id, Vector3 at, Vector3 axis, int turn)
        {
            var sign = new GameObject("Sign_" + id).transform;
            sign.SetParent(parent);
            sign.position = at + axis * 0.7f;
            sign.rotation = Quaternion.LookRotation(axis, Vector3.up);

            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Disc";
            disc.transform.SetParent(sign);
            disc.transform.localPosition = Vector3.zero;
            // A cylinder's axis is its local Y; tip it so the flat face points at the driver.
            disc.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            disc.transform.localScale = new Vector3(3.4f, 0.16f, 3.4f);
            Object.Destroy(disc.GetComponent<Collider>());
            Paint(disc, new Color(0.09f, 0.26f, 0.60f));

            // Rotation about local Z is counter-clockwise seen from +Z, so -90 sends the arrow to
            // local +X (the driver's left) and +90 to local -X (their right).
            //
            // The rightmost lane carries TWO movements, not one: intersection.con.xml connects
            // lane 0 to both the through edge and the right edge. A lone right arrow above it
            // describes a right-turn-only lane that does not exist, and makes a through-bound car
            // waiting there look like a stuck vehicle. Two arrows say what the lane actually is.
            if (turn == 2)
            {
                AddArrow(sign, 0f, new Vector3(0.80f, 0f, 0.30f), 0.95f);    // straight
                AddArrow(sign, 90f, new Vector3(-0.80f, 0f, 0.30f), 0.95f);  // right
            }
            else
            {
                AddArrow(sign, turn == 0 ? -90f : 0f, new Vector3(0f, 0f, 0.30f), 1.25f);
            }
        }

        /// <summary>One white arrow glyph on a sign face.</summary>
        /// <param name="offset">Local position. +Z, NOT -Z: local +Z faces the oncoming driver,
        /// and on the far face the arrow only reads from behind the sign, mirrored. A Unity
        /// cylinder is 2 units tall, so the disc's faces sit at local z = +-0.16; at +0.14 the
        /// arrow sat INSIDE the disc and z-fought it into a blue-and-white mottle. Clear the
        /// face.</param>
        private static void AddArrow(Transform sign, float angle, Vector3 offset, float scale)
        {
            var arrow = new GameObject("Arrow");
            arrow.transform.SetParent(sign);
            arrow.transform.localPosition = offset;
            arrow.transform.localRotation = Quaternion.Euler(0f, 0f, angle);
            arrow.transform.localScale = Vector3.one * scale;
            arrow.AddComponent<MeshFilter>().sharedMesh = ArrowMesh;
            arrow.AddComponent<MeshRenderer>();
            // Emissive, not just white: the arrow is a vertical face with a horizontal normal, so
            // the overhead key light barely grazes it and plain white paint renders mid-grey.
            arrow.GetComponent<Renderer>().sharedMaterial = ArrowMaterial;
        }

        private static Material _arrowMat;

        private static Material ArrowMaterial
        {
            get
            {
                if (_arrowMat != null) return _arrowMat;
                _arrowMat = new Material(LitShader) { color = Color.white };
                _arrowMat.EnableKeyword("_EMISSION");
                _arrowMat.SetColor("_EmissionColor", new Color(0.75f, 0.78f, 0.82f));
                return _arrowMat;
            }
        }

        private static Mesh _arrow;

        /// <summary>
        /// A flat arrow in local XY pointing +Y: shaft quad plus a triangular head.
        ///
        /// Built as one mesh rather than assembled from rotated cubes - three boxes at angles
        /// read as a wrench, not an arrow, because the barbs overshoot the shaft tip instead of
        /// meeting it. Double-sided with explicit normals so it lights correctly from either side.
        /// </summary>
        private static Mesh ArrowMesh
        {
            get
            {
                if (_arrow != null) return _arrow;

                const float w = 0.30f;   // half shaft width
                const float hw = 0.78f;  // half head width
                const float y0 = -1.05f; // shaft base
                const float y1 = 0.10f;  // shaft top / head base
                const float y2 = 1.10f;  // tip

                var face = new[]
                {
                    new Vector3(-w, y0, 0f), new Vector3(w, y0, 0f),
                    new Vector3(w, y1, 0f), new Vector3(-w, y1, 0f),
                    new Vector3(-hw, y1, 0f), new Vector3(hw, y1, 0f), new Vector3(0f, y2, 0f),
                };
                var faceTris = new[] { 0, 2, 1, 0, 3, 2, 4, 6, 5 };

                var verts = new Vector3[face.Length * 2];
                var norms = new Vector3[face.Length * 2];
                var tris = new int[faceTris.Length * 2];
                for (var i = 0; i < face.Length; i++)
                {
                    verts[i] = face[i];
                    verts[i + face.Length] = face[i];
                    norms[i] = Vector3.forward;
                    norms[i + face.Length] = Vector3.back;
                }
                for (var i = 0; i < faceTris.Length; i++) tris[i] = faceTris[i];
                for (var i = 0; i < faceTris.Length; i += 3)
                {
                    // Reversed winding for the back face.
                    tris[faceTris.Length + i] = faceTris[i] + face.Length;
                    tris[faceTris.Length + i + 1] = faceTris[i + 2] + face.Length;
                    tris[faceTris.Length + i + 2] = faceTris[i + 1] + face.Length;
                }

                _arrow = new Mesh { name = "Arrow", vertices = verts, normals = norms, triangles = tris };
                _arrow.RecalculateBounds();
                return _arrow;
            }
        }

        /// <summary>One lamp, sunk into the front face of the housing so it faces oncoming traffic.</summary>
        private static Renderer Lamp(Transform parent, string name, Vector3 at, float height, Vector3 axis)
        {
            var lamp = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            lamp.name = name;
            lamp.transform.SetParent(parent);
            lamp.transform.localScale = new Vector3(1.5f, 1.5f, 1.5f);
            // Nudge towards the approach - the traffic it governs comes from further out.
            lamp.transform.position = at + axis * 0.75f + Vector3.up * height;
            Object.Destroy(lamp.GetComponent<Collider>());
            return lamp.GetComponent<Renderer>();
        }

        /// <summary>Outward unit vector for approach 0=N, 1=E, 2=S, 3=W.</summary>
        private static Vector3 AxisOf(int approach)
        {
            switch (approach)
            {
                case 0: return Vector3.forward;   // +z, north
                case 1: return Vector3.right;     // +x, east
                case 2: return Vector3.back;      // -z, south
                default: return Vector3.left;     // -x, west
            }
        }

        /// <summary>A box sized <paramref name="across"/> x <paramref name="along"/> in the axis frame.</summary>
        private static Vector3 Size(Vector3 axis, float across, float along)
        {
            return Mathf.Abs(axis.z) > 0.5f
                ? new Vector3(across, 0.3f, along)
                : new Vector3(along, 0.3f, across);
        }

        private static void Marking(Transform parent, string name, Vector3 centre,
            float across, float along, Vector3 axis)
        {
            var strip = GameObject.CreatePrimitive(PrimitiveType.Cube);
            strip.name = name;
            strip.transform.SetParent(parent);
            strip.transform.localScale = Size(axis, across, along);
            strip.transform.position = new Vector3(centre.x, 0.07f, centre.z);
            Object.Destroy(strip.GetComponent<Collider>());
            Paint(strip, MarkingPaint);
        }

        private static GameObject Slab(GameObject root, string name, Vector3 size, Vector3 centre, Color color)
        {
            var slab = GameObject.CreatePrimitive(PrimitiveType.Cube);
            slab.name = name;
            slab.transform.SetParent(root.transform);
            slab.transform.localScale = size;
            slab.transform.position = centre;
            Object.Destroy(slab.GetComponent<Collider>());
            Paint(slab, color);
            return slab;
        }

        private static readonly Dictionary<Color, Material> Palette = new Dictionary<Color, Material>();

        /// <summary>
        /// Assigns a shared material for this colour, creating it once.
        ///
        /// Deliberately <c>sharedMaterial</c>, not <c>material</c>: the latter instantiates a
        /// private copy per renderer, and the scenery alone is ~1000 objects - that would be a
        /// thousand materials and a thousand un-batchable draw calls. Nothing recolours an object
        /// painted this way after the fact; the signal lamps own their materials separately.
        /// </summary>
        public static void Paint(GameObject go, Color color)
        {
            // A null shader means this build shipped none. Leaving the default material is ugly;
            // `new Material(null)` throws and takes the entire scene build down with it.
            if (LitShader == null) return;
            // `mat == null` also catches a Material destroyed when Play mode last exited: the
            // dictionary is static and survives that, so a stale entry would otherwise be handed
            // out on the next Play whenever domain reload is turned off.
            if (!Palette.TryGetValue(color, out var mat) || mat == null)
            {
                mat = new Material(LitShader) { color = color };
                Palette[color] = mat;
            }
            go.GetComponent<Renderer>().sharedMaterial = mat;
        }
    }
}
