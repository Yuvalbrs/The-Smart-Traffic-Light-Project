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
        /// <summary>No mountain footprint may come closer than this to the junction centre.
        /// The arms reach <c>IntersectionScene.ArmLength</c> (150 m); this leaves a clear
        /// margin beyond the far end of every one of them.</summary>
        private const float Clearance = 300f;

        /// <summary>How far out clouds must stay in plan, so none sits over the junction and
        /// blocks the top-down camera. The top-down view sees roughly 120 m either side of
        /// centre at its height, so this is comfortably outside the frame.</summary>
        private const float CloudClearance = 360f;

        private static readonly Color SnowColor = new Color(0.92f, 0.94f, 0.97f);
        /// <summary>The daylight sky, which distant geometry fades toward.</summary>
        private static readonly Color HazeColor = new Color(0.58f, 0.68f, 0.80f);

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
            BuildCelestialBody(root);
        }

        /// <summary>
        /// The sun by day, the moon by night - one sphere that DayNight recolours and moves.
        ///
        /// It is placed and sized entirely by <see cref="DayNight.Apply"/>, from the same rotation
        /// that aims the directional light, so the disc and the shadows can never disagree about
        /// where the light is coming from. Nothing here fixes a position; doing so would create a
        /// second source of truth for it.
        /// </summary>
        private static void BuildCelestialBody(Transform root)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = "CelestialBody";
            go.transform.SetParent(root);

            // CreatePrimitive ships a collider nobody asked for. Nothing in this viewer raycasts,
            // but a 110 m sphere of collider hanging over the scene is a trap for anything that
            // later does.
            var collider = go.GetComponent<Collider>();
            if (collider != null) UnityEngine.Object.Destroy(collider);

            var renderer = go.GetComponent<Renderer>();
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;

            DayNight.RegisterBody(go.transform, renderer);
        }

        /// <summary>
        /// Trees: a tapering trunk and a canopy built from lobes.
        ///
        /// Two stacked cones on a plain cylinder is a diagram of a tree. What the eye reads as a
        /// tree is a trunk that thins as it rises and a crown with more than one bulge in it, so
        /// the outline is not a triangle. Broadleaf crowns use the same union-of-lobes surface as
        /// the clouds, but FACETED rather than smooth - foliage catching light in planes is what
        /// makes low-poly greenery read as leaves instead of as a balloon.
        /// </summary>
        private static void BuildTrees(Transform root, System.Random rng)
        {
            var trees = new GameObject("Trees").transform;
            trees.SetParent(root);

            for (var i = 0; i < 260; i++)
            {
                var x = (float)(rng.NextDouble() * 2 - 1) * 260f;
                var z = (float)(rng.NextDouble() * 2 - 1) * 260f;

                // Keep clear of both road corridors and of the junction itself, with a verge so
                // nothing overhangs the carriageway.
                const float clear = IntersectionScene.HalfRoad + 8f;
                if (Mathf.Abs(x) < clear || Mathf.Abs(z) < clear) continue;

                var height = 7f + (float)rng.NextDouble() * 10f;
                var colour = Foliage[rng.Next(Foliage.Length)];
                var at = new Vector3(x, 0f, z);

                // Proportions are stated as fractions of the tree's HEIGHT, which is the fix.
                // They used to be multiples of the trunk radius, so a slightly fat trunk produced
                // a crown two-thirds as wide as the tree was tall, hanging down to a quarter of
                // its height - a bush with a stick under it. A tree is mostly clear trunk with
                // foliage in the top third.
                var trunkR = height * 0.035f;              // ~0.4 m on a 12 m tree
                var crownBottom = height * 0.42f;          // no green below here
                var trunkTop = height * 0.70f;             // ends inside the crown, never below it

                Taper(trees, at, trunkR * 1.35f, trunkR, trunkTop * 0.55f, 0f);
                Taper(trees, at, trunkR, trunkR * 0.78f, trunkTop * 0.45f, trunkTop * 0.55f);

                if (rng.NextDouble() < 0.45)
                {
                    // Radius follows from where the crown has to start and stop, rather than
                    // being picked and then hoping it lands well.
                    var crownR = (height - crownBottom) * (0.62f + (float)rng.NextDouble() * 0.18f);
                    var centreY = crownBottom + crownR;

                    var count = 4 + rng.Next(3);
                    var lobes = new Blob.Lobe[count];
                    // Lobe 0 sits on the trunk line so the crown always encloses it.
                    lobes[0] = new Blob.Lobe(Vector3.zero, crownR);
                    for (var l = 1; l < count; l++)
                    {
                        lobes[l] = new Blob.Lobe(
                            new Vector3((float)(rng.NextDouble() - 0.5) * crownR * 0.75f,
                                (float)(rng.NextDouble() - 0.5) * crownR * 0.5f,
                                (float)(rng.NextDouble() - 0.5) * crownR * 0.75f),
                            crownR * (0.60f + (float)rng.NextDouble() * 0.3f));
                    }

                    var crown = new GameObject("Crown");
                    crown.transform.SetParent(trees);
                    crown.transform.position = at + Vector3.up * centreY;
                    crown.transform.rotation = Quaternion.Euler(0f, (float)rng.NextDouble() * 360f, 0f);
                    crown.AddComponent<MeshFilter>().sharedMesh = Blob.Build(
                        lobes, rings: 8, sectors: 12, flatBase: -999f, smooth: false,
                        jitter: 0.14f, rng: rng);
                    crown.AddComponent<MeshRenderer>();
                    IntersectionScene.Paint(crown, colour);
                    continue;
                }

                // Conifer: three tiers in the top half, widest at the bottom. This shape was
                // already right in the build - the broadleaf is now measured the same way.
                var tierR = height * 0.20f;
                Cone(trees, "Fir", at + Vector3.up * crownBottom, tierR * 1.30f, height * 0.26f, colour);
                Cone(trees, "Fir", at + Vector3.up * (crownBottom + height * 0.16f), tierR * 1.00f, height * 0.24f, colour);
                Cone(trees, "Fir", at + Vector3.up * (crownBottom + height * 0.30f), tierR * 0.66f, height * 0.22f, colour);
            }
        }

        /// <summary>A trunk section that narrows as it rises.</summary>
        private static void Taper(Transform parent, Vector3 at, float rBottom, float rTop,
            float height, float yOffset)
        {
            var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            trunk.name = "Trunk";
            trunk.transform.SetParent(parent);
            var mid = (rBottom + rTop) * 0.5f;
            trunk.transform.localScale = new Vector3(mid * 2f, height / 2f, mid * 2f);
            trunk.transform.position = at + Vector3.up * (yOffset + height / 2f);
            Object.Destroy(trunk.GetComponent<Collider>());
            IntersectionScene.Paint(trunk, TrunkColor);
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
            BuildRidgeRing(hills, rng, count: 26, minDist: 620f, spread: 320f,
                minHeight: 130f, heightSpread: 150f, snowChance: 0.70f);
            BuildRidgeRing(hills, rng, count: 20, minDist: 420f, spread: 160f,
                minHeight: 55f, heightSpread: 65f, snowChance: 0.10f);
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
                    // Much wider than tall. The first version used 0.8-1.35x the height, which is
                    // a spire; hills in the world are several times wider than they are high, and
                    // that ratio is most of what makes a silhouette read as rock.
                    var radius = height * (1.5f + (float)rng.NextDouble() * 0.7f);
                    var yaw = (float)rng.NextDouble() * 360f;

                    // Keep the whole footprint off the network. A mountain centred beyond the
                    // arms can still overlap them once it is this wide - which is exactly what
                    // put one through the middle of a carriageway.
                    var here = at + jitter;
                    var flat = new Vector3(here.x, 0f, here.z);
                    var needed = Clearance + radius;
                    if (flat.magnitude < needed)
                    {
                        var dir = flat.sqrMagnitude > 0.01f ? flat.normalized : Vector3.forward;
                        here = new Vector3(dir.x * needed, here.y, dir.z * needed);
                    }

                    // One ridge drives both meshes, so the snow cannot drift off the rock.
                    const int sides = 26;
                    var ridge = new Ridge(rng);

                    // Aerial perspective: air between the viewer and the rock scatters light, so
                    // distant ridges wash toward the sky colour. It is the cue that separates a
                    // far wall from a near one, and without it every ridge sits at one depth.
                    var haze = Mathf.Clamp01((here.magnitude - 350f) / 900f);
                    var rock = Color.Lerp(HillColor[rng.Next(HillColor.Length)], HazeColor, haze * 0.72f);
                    Place(hills, "Mountain", here, yaw, radius, height,
                        RidgeMesh(ridge, sides, 0f, 1f, 1f, rng), rock);

                    if (rng.NextDouble() < snowChance * drop)
                    {
                        var snowline = 0.56f + (float)rng.NextDouble() * 0.18f;
                        Place(hills, "Snow", here, yaw, radius, height,
                            // 1.5% outward keeps it clear of the rock it sits on; any less and the
                            // two surfaces z-fight into a shimmering mess at this camera distance.
                            RidgeMesh(ridge, sides, snowline, 1f, 1.015f, rng),
                            Color.Lerp(SnowColor, HazeColor, haze * 0.45f));
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
                // Roughly half the relief of the first attempt. At 0.30 the base octave moved the
                // outline by nearly a third of the radius, which reads as a shard rather than as
                // a mountain; real ranges are mostly mass with detail on top, not detail.
                _amp = new[] { 0.17f, 0.085f, 0.042f, 0.020f };
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
            const int rings = 14;   // more rings = a smoother profile between base and summit

            // Ring 0..rings-1 of positions, then the apex; triangles copy from this grid.
            var grid = new Vector3[rings, sides];
            for (var r = 0; r < rings; r++)
            {
                var v = Mathf.Lerp(vFrom, vTo, r / (float)(rings - 1));
                // Gentler than the 1.42 first used, and it never reaches zero: a profile that
                // closes to a point puts a needle on top of every summit. Real peaks are blunt,
                // and the 0.12 floor is what turns the spike into a summit ridge.
                var ringR = Mathf.Lerp(Mathf.Pow(1f - v, 1.12f), 0.12f, v * v * 0.55f);
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
        /// Cumulus: ONE lumpy surface per cloud, with a flat bottom.
        ///
        /// The previous version scattered separate ellipsoids around a point, and separate
        /// ellipsoids do not make a cloud - they make a pile of balls, with a visible outline
        /// wherever one ends and the next begins. That is what it looked like, and no amount of
        /// tuning the placement was going to fix it, because the problem was that the cloud was
        /// several objects.
        ///
        /// It is now a single mesh: the lobes are unioned into one surface by <see cref="Blob"/>,
        /// so there are no internal edges at all. Smooth normals rather than facets, because a
        /// cloud is the one thing in this scene that should not look faceted.
        ///
        /// The base is flat because that is physically why cumulus have flat bottoms - it is the
        /// altitude where rising air reaches its condensation temperature, and every cloud in one
        /// air mass finds it at the same height. A fully round cloud reads as a balloon.
        /// </summary>
        private static void BuildClouds(Transform root, System.Random rng)
        {
            var clouds = new GameObject("Clouds").transform;
            clouds.SetParent(root);

            for (var i = 0; i < 26; i++)
            {
                var scale = 26f + (float)rng.NextDouble() * 26f;
                var drift = (float)(rng.NextDouble() * Mathf.PI * 2);

                // Lobes along one axis, fattest in the middle, all sitting on the same floor.
                var count = 4 + rng.Next(4);
                var lobes = new Blob.Lobe[count];
                for (var l = 0; l < count; l++)
                {
                    var t = count == 1 ? 0f : l / (float)(count - 1) * 2f - 1f;   // -1..1
                    var bulge = 1f - 0.55f * Mathf.Abs(t);
                    lobes[l] = new Blob.Lobe(
                        new Vector3(t * scale * 0.85f,
                            (float)rng.NextDouble() * scale * 0.20f,
                            (float)(rng.NextDouble() - 0.5) * scale * 0.35f),
                        scale * (0.48f + 0.32f * bulge));
                }

                var mesh = Blob.Build(lobes, rings: 14, sectors: 22,
                    flatBase: -scale * 0.16f, smooth: true, jitter: 0.06f, rng: rng);

                // Never directly over the network. The top-down preset sits at about 210 m and
                // the clouds live at 120-190 m, so anything above the junction is BETWEEN the
                // camera and the road - one cloud was covering a third of that view. Keeping the
                // column clear is the fix; raising the layer above every camera would only make
                // them invisible from the ground.
                var cx = (float)(rng.NextDouble() * 2 - 1) * 620f;
                var cz = (float)(rng.NextDouble() * 2 - 1) * 620f;
                var flat = new Vector2(cx, cz);
                if (flat.magnitude < CloudClearance)
                {
                    var dir = flat.sqrMagnitude > 0.01f ? flat.normalized : Vector2.up;
                    cx = dir.x * CloudClearance;
                    cz = dir.y * CloudClearance;
                }

                var go = new GameObject("Cloud");
                go.transform.SetParent(clouds);
                go.transform.position = new Vector3(cx, 120f + (float)rng.NextDouble() * 70f, cz);
                go.transform.rotation = Quaternion.Euler(0f, drift * Mathf.Rad2Deg, 0f);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>();
                IntersectionScene.Paint(go, CloudColor);
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
