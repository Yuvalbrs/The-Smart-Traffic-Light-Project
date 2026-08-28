// T-05-03 - low-poly surroundings.
//
// Purely cosmetic: nothing here is read from the simulation, and nothing the simulation reports
// depends on it. It earns its place by giving the eye a horizon and a sense of scale - without
// it a 300 m network on a flat plane reads as an abstract diagram, and in a defence demo the
// audience should be able to see at a glance that this is a road.
//
// Everything is generated, so there are no imported assets, no licences and no meshes in git.
// Unity has no cone primitive, so the firs and the hills come from a generated cone mesh.

using UnityEngine;

namespace SmartTraffic
{
    public static class Scenery
    {
        // Deterministic: the scene must look identical every run, so a screenshot in the report
        // matches what the examiners see on the day.
        private const int Seed = 20260828;

        private static readonly Color TrunkColor = new Color(0.32f, 0.23f, 0.16f);
        private static readonly Color[] Foliage =
        {
            new Color(0.13f, 0.35f, 0.22f), new Color(0.17f, 0.45f, 0.26f),
            new Color(0.22f, 0.55f, 0.30f), new Color(0.30f, 0.62f, 0.32f),
        };
        private static readonly Color[] HillColor =
        {
            new Color(0.28f, 0.42f, 0.34f), new Color(0.34f, 0.47f, 0.52f),
            new Color(0.45f, 0.55f, 0.62f),
        };
        private static readonly Color SnowColor = new Color(0.92f, 0.94f, 0.97f);
        private static readonly Color CloudColor = new Color(0.96f, 0.97f, 1.00f);

        private static Mesh _cone;

        /// <summary>A low-poly cone, apex at +y, unit radius and height. Cached.</summary>
        private static Mesh ConeMesh
        {
            get
            {
                    // Same staleness guard as the material palette: this static outlives Play mode
                // but the Mesh it points to does not.
                if (_cone != null) return _cone;
                const int sides = 7; // odd count keeps the silhouette from looking machined
                var verts = new Vector3[sides + 2];
                verts[0] = new Vector3(0f, 1f, 0f);       // apex
                verts[sides + 1] = Vector3.zero;          // base centre
                for (var i = 0; i < sides; i++)
                {
                    var a = i / (float)sides * Mathf.PI * 2f;
                    verts[i + 1] = new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a));
                }

                var tris = new int[sides * 6];
                var t = 0;
                for (var i = 0; i < sides; i++)
                {
                    var cur = i + 1;
                    var next = i + 2 > sides ? 1 : i + 2;
                    tris[t++] = 0; tris[t++] = next; tris[t++] = cur;                 // side
                    tris[t++] = sides + 1; tris[t++] = cur; tris[t++] = next;         // base
                }

                _cone = new Mesh { name = "LowPolyCone", vertices = verts, triangles = tris };
                _cone.RecalculateNormals();
                _cone.RecalculateBounds();
                return _cone;
            }
        }

        public static void Build(Transform parent)
        {
            var root = new GameObject("Scenery").transform;
            root.SetParent(parent);

            var rng = new System.Random(Seed);
            BuildTrees(root, rng);
            BuildHills(root, rng);
            BuildClouds(root, rng);
        }

        private static void BuildTrees(Transform root, System.Random rng)
        {
            var trees = new GameObject("Trees").transform;
            trees.SetParent(root);

            for (var i = 0; i < 320; i++)
            {
                var x = (float)(rng.NextDouble() * 2 - 1) * 260f;
                var z = (float)(rng.NextDouble() * 2 - 1) * 260f;

                // Keep clear of both road corridors and of the junction itself, with a verge so
                // nothing overhangs the carriageway.
                const float clear = IntersectionScene.HalfRoad + 8f;
                if (Mathf.Abs(x) < clear || Mathf.Abs(z) < clear) continue;

                var height = 6f + (float)rng.NextDouble() * 9f;
                var radius = height * (0.22f + (float)rng.NextDouble() * 0.08f);
                var colour = Foliage[rng.Next(Foliage.Length)];
                var at = new Vector3(x, 0f, z);

                Cone(trees, "Fir", at + Vector3.up * (height * 0.35f), radius, height * 0.75f, colour);
                Cone(trees, "Fir", at + Vector3.up * (height * 0.70f), radius * 0.72f, height * 0.55f, colour);
                var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                trunk.name = "Trunk";
                trunk.transform.SetParent(trees);
                trunk.transform.localScale = new Vector3(radius * 0.22f, height * 0.2f, radius * 0.22f);
                trunk.transform.position = at + Vector3.up * (height * 0.2f);
                Object.Destroy(trunk.GetComponent<Collider>());
                IntersectionScene.Paint(trunk, TrunkColor);
            }
        }

        private static void BuildHills(Transform root, System.Random rng)
        {
            var hills = new GameObject("Hills").transform;
            hills.SetParent(root);

            for (var i = 0; i < 22; i++)
            {
                // Ring them well beyond the network so they read as distance, not obstacles.
                var angle = (float)(rng.NextDouble() * Mathf.PI * 2);
                var dist = 420f + (float)rng.NextDouble() * 260f;
                var at = new Vector3(Mathf.Cos(angle) * dist, -8f, Mathf.Sin(angle) * dist);

                var height = 90f + (float)rng.NextDouble() * 150f;
                var radius = height * (0.8f + (float)rng.NextDouble() * 0.5f);
                Cone(hills, "Hill", at, radius, height, HillColor[rng.Next(HillColor.Length)]);

                if (height > 170f) // snow cap on the tall ones only
                {
                    Cone(hills, "Snow", at + Vector3.up * (height * 0.72f),
                        radius * 0.29f, height * 0.28f, SnowColor);
                }
            }
        }

        private static void BuildClouds(Transform root, System.Random rng)
        {
            var clouds = new GameObject("Clouds").transform;
            clouds.SetParent(root);

            for (var i = 0; i < 26; i++)
            {
                var centre = new Vector3(
                    (float)(rng.NextDouble() * 2 - 1) * 500f,
                    150f + (float)rng.NextDouble() * 90f,
                    (float)(rng.NextDouble() * 2 - 1) * 500f);

                var puffs = 3 + rng.Next(3);
                for (var p = 0; p < puffs; p++)
                {
                    var puff = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    puff.name = "Cloud";
                    puff.transform.SetParent(clouds);
                    var size = 22f + (float)rng.NextDouble() * 26f;
                    puff.transform.localScale = new Vector3(size, size * 0.55f, size * 0.8f);
                    puff.transform.position = centre + new Vector3(
                        (float)(rng.NextDouble() * 2 - 1) * 26f, (float)rng.NextDouble() * 6f,
                        (float)(rng.NextDouble() * 2 - 1) * 16f);
                    Object.Destroy(puff.GetComponent<Collider>());
                    IntersectionScene.Paint(puff, CloudColor);
                }
            }
        }

        private static void Cone(Transform parent, string name, Vector3 at, float radius, float height, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent);
            go.transform.position = at;
            go.transform.localScale = new Vector3(radius, height, radius);
            go.AddComponent<MeshFilter>().sharedMesh = ConeMesh;
            go.AddComponent<MeshRenderer>();
            IntersectionScene.Paint(go, color);
        }
    }
}
