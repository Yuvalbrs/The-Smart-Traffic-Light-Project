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
        //: The underside of a cumulus is never white. Two flat shades fake self-shadowing
        //: cheaply; a real gradient would need a shader this build deliberately does not have.
        private static readonly Color CloudShadow = new Color(0.78f, 0.81f, 0.88f);

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

                // Split vertices, for the same reason the mountains use them: shared corners plus
                // RecalculateNormals averages the normals and shades the cone smoothly, so a
                // seven-sided fir came out looking like a smooth green blob instead of showing
                // its seven faces.
                var apex = new Vector3(0f, 1f, 0f);
                var centre = Vector3.zero;
                var verts = new System.Collections.Generic.List<Vector3>(sides * 6);
                var tris = new System.Collections.Generic.List<int>(sides * 6);

                void Tri(Vector3 a, Vector3 b, Vector3 c)
                {
                    tris.Add(verts.Count); verts.Add(a);
                    tris.Add(verts.Count); verts.Add(b);
                    tris.Add(verts.Count); verts.Add(c);
                }

                Vector3 Rim(int i)
                {
                    var a = i / (float)sides * Mathf.PI * 2f;
                    return new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a));
                }

                for (var i = 0; i < sides; i++)
                {
                    var cur = Rim(i);
                    var next = Rim((i + 1) % sides);
                    Tri(apex, next, cur);        // side
                    Tri(centre, cur, next);      // base
                }

                _cone = new Mesh { name = "LowPolyCone" };
                _cone.SetVertices(verts);
                _cone.SetTriangles(tris, 0);
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

                var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                trunk.name = "Trunk";
                trunk.transform.SetParent(trees);
                trunk.transform.localScale = new Vector3(radius * 0.22f, height * 0.2f, radius * 0.22f);
                trunk.transform.position = at + Vector3.up * (height * 0.2f);
                Object.Destroy(trunk.GetComponent<Collider>());
                IntersectionScene.Paint(trunk, TrunkColor);

                // A third of them are broadleaf. A verge of nothing but identical firs reads as
                // wallpaper; mixing two crown shapes is the cheapest way to break that up, and
                // roadside planting is mixed in any case.
                if (rng.NextDouble() < 0.34)
                {
                    var crowns = 2 + rng.Next(2);
                    for (var c = 0; c < crowns; c++)
                    {
                        var blob = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                        blob.name = "Crown";
                        blob.transform.SetParent(trees);
                        var s = radius * (1.5f + (float)rng.NextDouble() * 0.7f);
                        blob.transform.localScale = new Vector3(s, s * 0.85f, s);
                        blob.transform.position = at + Vector3.up * (height * 0.55f)
                            + new Vector3((float)(rng.NextDouble() - 0.5) * radius * 1.3f,
                                (float)rng.NextDouble() * height * 0.22f,
                                (float)(rng.NextDouble() - 0.5) * radius * 1.3f);
                        Object.Destroy(blob.GetComponent<Collider>());
                        IntersectionScene.Paint(blob, colour);
                    }
                    continue;
                }

                Cone(trees, "Fir", at + Vector3.up * (height * 0.35f), radius, height * 0.75f, colour);
                Cone(trees, "Fir", at + Vector3.up * (height * 0.70f), radius * 0.72f, height * 0.55f, colour);
            }
        }

        /// <summary>
        /// Ridged mountains, each its own generated mesh rather than a scaled cone.
        ///
        /// A cone is rotationally symmetric, so a ring of them reads as a row of identical party
        /// hats however the colours are varied - which is exactly what made the old horizon look
        /// like a diagram. Displacing each ring of vertices by its own radial noise breaks that
        /// symmetry: every mountain gets spurs and gullies, no two silhouettes repeat, and the
        /// profile is concave near the base (mountains flare out at the bottom) instead of
        /// straight-sided.
        ///
        /// The snow cap is a second mesh built from the SAME ridge profile, covering only the
        /// rings above the snowline and inflated by a hair to sit on the rock. So it follows the
        /// spurs and gullies down instead of being a smooth white cone balanced on the summit.
        /// Two meshes rather than one with per-vertex colour, because the pinned shader is
        /// Standard and Standard ignores <c>mesh.colors</c> entirely - vertex colours would have
        /// compiled, run, and rendered a uniformly white mountain range.
        /// </summary>
        private static void BuildHills(Transform root, System.Random rng)
        {
            var hills = new GameObject("Hills").transform;
            hills.SetParent(root);

            // Two staggered rings: a far wall and a nearer, lower one. A single ring at one radius
            // gives every peak the same apparent size, which flattens the horizon.
            BuildRidgeRing(hills, rng, count: 26, minDist: 520f, spread: 240f,
                minHeight: 150f, heightSpread: 190f, snowChance: 0.75f);
            BuildRidgeRing(hills, rng, count: 18, minDist: 380f, spread: 120f,
                minHeight: 70f, heightSpread: 80f, snowChance: 0.12f);
        }

        private static void BuildRidgeRing(Transform hills, System.Random rng, int count,
            float minDist, float spread, float minHeight, float heightSpread, float snowChance)
        {
            for (var i = 0; i < count; i++)
            {
                // Even angular spacing plus jitter: pure random angles clump and leave gaps, and a
                // gap on the horizon looks like a missing object rather than a valley.
                var angle = (i / (float)count) * Mathf.PI * 2f
                            + (float)(rng.NextDouble() - 0.5) * (Mathf.PI * 2f / count) * 0.8f;
                var dist = minDist + (float)rng.NextDouble() * spread;
                var at = new Vector3(Mathf.Cos(angle) * dist, -10f, Mathf.Sin(angle) * dist);

                // Each position gets a small CLUSTER rather than one peak. A range is overlapping
                // massifs with subsidiary summits; one freestanding cone per slot is what made the
                // horizon read as a row of hats no matter how the silhouette was shaped.
                var peaks = 1 + rng.Next(3);
                for (var k = 0; k < peaks; k++)
                {
                    var jitter = new Vector3(
                        (float)(rng.NextDouble() - 0.5) * 130f, 0f,
                        (float)(rng.NextDouble() - 0.5) * 130f);
                    // Subsidiary peaks are lower, which is what makes the main one read as main.
                    var drop = k == 0 ? 1f : 0.50f + (float)rng.NextDouble() * 0.28f;
                    var height = (minHeight + (float)rng.NextDouble() * heightSpread) * drop;
                    var radius = height * (0.80f + (float)rng.NextDouble() * 0.55f);
                    var yaw = (float)rng.NextDouble() * 360f;

                    // One ridge drives both meshes, so the snow cannot drift off the rock.
                    const int sides = 26;
                    var ridge = new Ridge(rng);

                    Place(hills, "Mountain", at + jitter, yaw, radius, height,
                        RidgeMesh(ridge, sides, 0f, 1f, 1f, rng),
                        HillColor[rng.Next(HillColor.Length)]);

                    if (rng.NextDouble() < snowChance * drop)
                    {
                        var snowline = 0.56f + (float)rng.NextDouble() * 0.18f;
                        Place(hills, "Snow", at + jitter, yaw, radius, height,
                            // 1.5% outward keeps it clear of the rock it sits on; any less and the
                            // two surfaces z-fight into a shimmering mess at this camera distance.
                            RidgeMesh(ridge, sides, snowline, 1f, 1.015f, rng),
                            SnowColor);
                    }
                }
            }
        }

        private static void Place(Transform parent, string name, Vector3 at, float yaw,
            float radius, float height, Mesh mesh, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent);
            go.transform.position = at;
            go.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            go.transform.localScale = new Vector3(radius, height, radius);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>();
            IntersectionScene.Paint(go, color);
        }

        /// <summary>
        /// The silhouette of one mountain: a periodic sum of sines around the compass.
        ///
        /// Integer frequencies are what make it periodic - theta and theta+2pi must give the same
        /// radius or the mesh splits open along the seam. Several octaves at falling amplitude is
        /// the standard fBm recipe, and it is the difference between a lumpy cone and something
        /// with a big shoulder, a couple of spurs off it, and fine broken detail on top. A single
        /// octave - what the first version used - just makes a cone slightly oval.
        /// </summary>
        private sealed class Ridge
        {
            private readonly int[] _freq;
            private readonly float[] _amp, _phase;

            public Ridge(System.Random rng)
            {
                _freq = new[] { 2 + rng.Next(2), 5 + rng.Next(3), 11 + rng.Next(5), 19 + rng.Next(7) };
                _amp = new[] { 0.30f, 0.16f, 0.075f, 0.035f };
                _phase = new float[4];
                for (var i = 0; i < 4; i++) _phase[i] = (float)(rng.NextDouble() * Mathf.PI * 2);
            }

            public float At(float theta)
            {
                var s = 0f;
                for (var i = 0; i < _freq.Length; i++) s += _amp[i] * Mathf.Sin(_freq[i] * theta + _phase[i]);
                return s;
            }
        }

        /// <summary>
        /// A slice of one mountain: unit radius, unit height, apex at +y.
        ///
        /// Rings between <paramref name="vFrom"/> and <paramref name="vTo"/> (0 = base, 1 =
        /// summit) on a concave profile, displaced by the ridge function. The displacement is a
        /// function of DIRECTION only, so it is identical on every ring - that vertical coherence
        /// is what makes a spur read as a spur running down the mountain rather than as dents.
        ///
        /// Vertices are SPLIT: every triangle carries its own three, so each gets the face normal
        /// and the surface shades as flat facets. Sharing vertices and calling RecalculateNormals
        /// averages the normals at every corner, which is smooth shading - and a smooth-shaded
        /// low-poly mountain looks like a melted blob, which is exactly how the first attempt
        /// came out.
        ///
        /// <paramref name="inflate"/> scales the slice outward, to lift the snow clear of the
        /// rock it shares a surface with.
        /// </summary>
        private static Mesh RidgeMesh(Ridge ridge, int sides, float vFrom, float vTo,
            float inflate, System.Random rng)
        {
            const int rings = 9;

            // Ring 0..rings-1 of positions, then the apex; triangles copy from this grid.
            var grid = new Vector3[rings, sides];
            for (var r = 0; r < rings; r++)
            {
                var v = Mathf.Lerp(vFrom, vTo, r / (float)(rings - 1));
                // Concave profile - a wide skirt and steep shoulders, not a cone's straight sides.
                var ringR = Mathf.Pow(1f - v, 1.42f);
                for (var s = 0; s < sides; s++)
                {
                    var a = s / (float)sides * Mathf.PI * 2f;
                    // Relief flattens towards the summit so the peak stays a peak, and a little
                    // vertical jitter stops the rings reading as contour lines.
                    var relief = 1f + ridge.At(a) * Mathf.Lerp(1f, 0.25f, v);
                    var rr = ringR * relief * inflate;
                    var yj = v + (float)(rng.NextDouble() - 0.5) * 0.012f;
                    grid[r, s] = new Vector3(Mathf.Cos(a) * rr, yj, Mathf.Sin(a) * rr);
                }
            }
            var apex = new Vector3(0f, vTo, 0f);

            var verts = new System.Collections.Generic.List<Vector3>((rings * sides) * 6);
            var tris = new System.Collections.Generic.List<int>((rings * sides) * 6);

            void Tri(Vector3 a, Vector3 b, Vector3 c)
            {
                tris.Add(verts.Count); verts.Add(a);
                tris.Add(verts.Count); verts.Add(b);
                tris.Add(verts.Count); verts.Add(c);
            }

            for (var r = 0; r < rings - 1; r++)
            {
                for (var s = 0; s < sides; s++)
                {
                    var s1 = (s + 1) % sides;
                    Tri(grid[r, s], grid[r + 1, s], grid[r, s1]);
                    Tri(grid[r, s1], grid[r + 1, s], grid[r + 1, s1]);
                }
            }
            for (var s = 0; s < sides; s++)
            {
                Tri(grid[rings - 1, s], apex, grid[rings - 1, (s + 1) % sides]);
            }

            var mesh = new Mesh { name = "Ridge" };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();   // per-triangle now, because nothing is shared
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>
        /// Cumulus, built as clusters of ellipsoids with a flat base.
        ///
        /// The old version scattered a handful of equal spheres around a point, which reads as a
        /// bunch of grapes: real cumulus sit ON a level, because that level is where the air
        /// reaches its condensation temperature. So every puff in a cloud shares one base height
        /// and only grows upward, the puffs get smaller the higher they sit, and the whole cluster
        /// is stretched along one axis instead of being spherical in plan.
        ///
        /// Two shades: the lower puffs are slightly grey, the upper ones near-white, which fakes
        /// the self-shadowing that makes a cloud look like it has volume.
        /// </summary>
        private static void BuildClouds(Transform root, System.Random rng)
        {
            var clouds = new GameObject("Clouds").transform;
            clouds.SetParent(root);

            for (var i = 0; i < 22; i++)
            {
                var baseY = 160f + (float)rng.NextDouble() * 80f;
                var centre = new Vector3(
                    (float)(rng.NextDouble() * 2 - 1) * 520f,
                    baseY,
                    (float)(rng.NextDouble() * 2 - 1) * 520f);

                // One horizontal direction the cluster elongates along - clouds are drawn out by
                // wind, not round.
                var drift = (float)(rng.NextDouble() * Mathf.PI * 2);
                var along = new Vector3(Mathf.Cos(drift), 0f, Mathf.Sin(drift));
                var across = new Vector3(-along.z, 0f, along.x);
                var spread = 30f + (float)rng.NextDouble() * 34f;
                var scale = 0.8f + (float)rng.NextDouble() * 0.7f;

                var puffs = 6 + rng.Next(5);
                for (var p = 0; p < puffs; p++)
                {
                    // Bias along the drift axis; a little across; upward only.
                    var u = (float)(rng.NextDouble() * 2 - 1);
                    var lift = (1f - Mathf.Abs(u)) * (float)rng.NextDouble();   // fattest in the middle
                    var at = centre
                             + along * (u * spread)
                             + across * ((float)(rng.NextDouble() * 2 - 1) * spread * 0.34f)
                             + Vector3.up * (lift * 15f * scale);

                    var size = (16f + (float)rng.NextDouble() * 16f) * scale
                               * Mathf.Lerp(1f, 0.62f, lift);   // smaller the higher it sits
                    var puff = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    puff.name = "Puff";
                    puff.transform.SetParent(clouds);
                    puff.transform.localScale = new Vector3(size, size * 0.62f, size * 0.86f);
                    puff.transform.position = at;
                    puff.transform.rotation = Quaternion.Euler(0f, drift * Mathf.Rad2Deg, 0f);
                    Object.Destroy(puff.GetComponent<Collider>());
                    IntersectionScene.Paint(puff, lift > 0.45f ? CloudColor : CloudShadow);
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
