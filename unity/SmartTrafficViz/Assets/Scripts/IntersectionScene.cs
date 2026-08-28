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
        public const float StopLine = HalfRoad + 1.4f;                 // signal heads just outside

        /// <summary>Lane centre offsets from the centreline, index 0 = rightmost = furthest out.</summary>
        private static readonly float[] LaneOffsets = { 8.0f, 4.8f, 1.6f };

        /// <summary>M0..M11 in the canonical N,E,S,W x (left, through, right) order.</summary>
        public static readonly string[] MovementIds =
        {
            "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11",
        };

        public static readonly Color Green = new Color(0.20f, 0.80f, 0.35f);
        public static readonly Color Yellow = new Color(0.95f, 0.80f, 0.20f);
        public static readonly Color Red = new Color(0.90f, 0.25f, 0.22f);

        public static Color ColorFor(string name)
        {
            switch (name)
            {
                case "green": return Green;
                case "yellow": return Yellow;
                default: return Red;
            }
        }

        /// <summary>Builds ground, roads and junction. Returns the root object.</summary>
        public static GameObject BuildStatic()
        {
            var root = new GameObject("Intersection");

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(root.transform);
            ground.transform.localScale = new Vector3(ArmLength / 5f, 1f, ArmLength / 5f);
            ground.transform.position = new Vector3(0f, -0.05f, 0f);
            Paint(ground, new Color(0.13f, 0.15f, 0.17f));

            var asphalt = new Color(0.28f, 0.29f, 0.31f);
            var junction = MakeSlab("Junction", new Vector3(RoadWidth, 0.1f, RoadWidth), Vector3.zero, asphalt);
            junction.transform.SetParent(root.transform);

            // One slab per arm, from the junction edge out to the approach endpoint.
            var armLen = ArmLength - HalfRoad;
            var armMid = HalfRoad + armLen / 2f;
            AddArm(root, "Arm_N", new Vector3(RoadWidth, 0.1f, armLen), new Vector3(0f, 0f, armMid), asphalt);
            AddArm(root, "Arm_S", new Vector3(RoadWidth, 0.1f, armLen), new Vector3(0f, 0f, -armMid), asphalt);
            AddArm(root, "Arm_E", new Vector3(armLen, 0.1f, RoadWidth), new Vector3(armMid, 0f, 0f), asphalt);
            AddArm(root, "Arm_W", new Vector3(armLen, 0.1f, RoadWidth), new Vector3(-armMid, 0f, 0f), asphalt);

            return root;
        }

        /// <summary>
        /// Creates the twelve signal heads, one per movement, at its own stop line.
        /// Returns them keyed by movement id so the renderer can recolour in place.
        /// </summary>
        public static Dictionary<string, Renderer> BuildSignalHeads(Transform parent)
        {
            var heads = new Dictionary<string, Renderer>();
            for (var approach = 0; approach < 4; approach++)
            {
                for (var turn = 0; turn < 3; turn++)
                {
                    // turn 0 = left = leftmost lane = index 2 = nearest the centreline.
                    var offset = LaneOffsets[2 - turn];
                    var id = MovementIds[approach * 3 + turn];
                    var head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    head.name = "Signal_" + id;
                    head.transform.SetParent(parent);
                    head.transform.localScale = Vector3.one * 1.8f;
                    head.transform.position = HeadPosition(approach, offset);
                    heads[id] = head.GetComponent<Renderer>();
                }
            }
            return heads;
        }

        /// <summary>
        /// Where an approach's signal head sits. Each approach occupies the half of its arm that
        /// is to the right of its own travel direction, which is what decides the sign here.
        /// </summary>
        private static Vector3 HeadPosition(int approach, float laneOffset)
        {
            const float h = 3.2f;
            switch (approach)
            {
                case 0: return new Vector3(-laneOffset, h, StopLine);   // N approach, heading south
                case 1: return new Vector3(StopLine, h, laneOffset);    // E approach, heading west
                case 2: return new Vector3(laneOffset, h, -StopLine);   // S approach, heading north
                default: return new Vector3(-StopLine, h, -laneOffset); // W approach, heading east
            }
        }

        private static void AddArm(GameObject root, string name, Vector3 size, Vector3 pos, Color color)
        {
            var arm = MakeSlab(name, size, pos, color);
            arm.transform.SetParent(root.transform);
        }

        private static GameObject MakeSlab(string name, Vector3 size, Vector3 centre, Color color)
        {
            var slab = GameObject.CreatePrimitive(PrimitiveType.Cube);
            slab.name = name;
            slab.transform.localScale = size;
            slab.transform.position = centre;
            Paint(slab, color);
            return slab;
        }

        private static void Paint(GameObject go, Color color)
        {
            var renderer = go.GetComponent<Renderer>();
            // Shader.Find at runtime rather than a serialised material: keeps the project free of
            // .mat assets, and falls back cleanly whichever render pipeline the editor is set to.
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            renderer.material = new Material(shader) { color = color };
        }
    }
}
