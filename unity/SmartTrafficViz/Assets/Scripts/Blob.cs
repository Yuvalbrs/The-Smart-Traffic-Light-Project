// One lumpy, continuous surface from several overlapping lobes.
//
// This exists because scattering separate spheres does not make a cloud - it makes a pile of
// spheres, and that is exactly what it looked like. What reads as a cloud (or a tree canopy) is a
// SINGLE surface with bulges: no internal outlines, no place where one ball visibly ends and the
// next begins.
//
// The usual way to get that is metaballs through marching cubes. That is a lot of machinery, and
// most of it is wasted here because these shapes are all star-shaped about their own centre - no
// overhangs, no holes. So instead of walking a voxel grid, the surface is sampled RADIALLY: fire a
// ray from the centre in each direction, ask each lobe where that ray leaves it, and keep the
// furthest. Union of spheres, one cheap closed mesh, no grid.
//
// The joins are softened with the standard polynomial smooth-max, because a hard max leaves a
// visible crease exactly where two lobes meet - which would put the outlines straight back.

using UnityEngine;

namespace SmartTraffic
{
    public static class Blob
    {
        public struct Lobe
        {
            public Vector3 Centre;
            public float Radius;

            public Lobe(Vector3 centre, float radius) { Centre = centre; Radius = radius; }
        }

        /// <summary>
        /// Builds the surface around the origin.
        /// </summary>
        /// <param name="lobes">Overlapping spheres, in local space. Each must contain the origin's
        /// general direction - they are unioned as seen FROM the origin.</param>
        /// <param name="rings">Latitude steps. More = rounder.</param>
        /// <param name="sectors">Longitude steps.</param>
        /// <param name="flatBase">Vertices below this local y are pulled up to it. Cumulus sit on
        /// a flat bottom because that is the altitude where the air condenses; a fully round cloud
        /// reads as a balloon. Pass a large negative number to leave it alone.</param>
        /// <param name="smooth">Smooth normals (soft, for cloud) or flat facets (for foliage).</param>
        /// <param name="jitter">Per-vertex radial noise, as a fraction. Breaks up the lat/long
        /// regularity that otherwise shows as banding.</param>
        public static Mesh Build(Lobe[] lobes, int rings, int sectors, float flatBase, bool smooth,
            float jitter, System.Random rng)
        {
            var grid = new Vector3[rings + 1, sectors];
            for (var r = 0; r <= rings; r++)
            {
                // Latitude from +y (0) to -y (pi).
                var phi = Mathf.PI * r / rings;
                var sy = Mathf.Cos(phi);
                var sr = Mathf.Sin(phi);
                for (var s = 0; s < sectors; s++)
                {
                    var theta = 2f * Mathf.PI * s / sectors;
                    var dir = new Vector3(sr * Mathf.Cos(theta), sy, sr * Mathf.Sin(theta));
                    var radius = SurfaceRadius(dir, lobes);
                    if (jitter > 0f) radius *= 1f + (float)(rng.NextDouble() - 0.5) * jitter;
                    var p = dir * radius;
                    if (p.y < flatBase) p.y = flatBase;
                    grid[r, s] = p;
                }
            }

            var mesh = new Mesh { name = "Blob" };
            if (smooth) BuildShared(mesh, grid, rings, sectors);
            else BuildSplit(mesh, grid, rings, sectors);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>How far the surface is from the centre along <paramref name="dir"/>.</summary>
        private static float SurfaceRadius(Vector3 dir, Lobe[] lobes)
        {
            var best = 0f;
            foreach (var lobe in lobes)
            {
                // Ray-sphere from the origin: the far intersection is d + sqrt(R^2 - (|c|^2 - d^2)).
                var d = Vector3.Dot(lobe.Centre, dir);
                var h2 = lobe.Radius * lobe.Radius - (lobe.Centre.sqrMagnitude - d * d);
                if (h2 <= 0f) continue;                 // this ray misses this lobe entirely
                var hit = d + Mathf.Sqrt(h2);
                if (hit <= 0f) continue;                 // lobe is entirely behind the centre
                best = best <= 0f ? hit : SmoothMax(best, hit, lobe.Radius * 0.55f);
            }
            return best;
        }

        /// <summary>Polynomial smooth maximum. A hard max creases where two lobes meet.</summary>
        private static float SmoothMax(float a, float b, float k)
        {
            if (k <= 0.0001f) return Mathf.Max(a, b);
            var h = Mathf.Clamp01(0.5f + 0.5f * (a - b) / k);
            return Mathf.Lerp(b, a, h) + k * h * (1f - h);
        }

        private static void BuildShared(Mesh mesh, Vector3[,] grid, int rings, int sectors)
        {
            var verts = new Vector3[(rings + 1) * sectors];
            var n = 0;
            for (var r = 0; r <= rings; r++)
                for (var s = 0; s < sectors; s++)
                    verts[n++] = grid[r, s];

            var tris = new System.Collections.Generic.List<int>(rings * sectors * 6);
            for (var r = 0; r < rings; r++)
            {
                for (var s = 0; s < sectors; s++)
                {
                    var s1 = (s + 1) % sectors;
                    var a = r * sectors + s;
                    var b = r * sectors + s1;
                    var c = (r + 1) * sectors + s;
                    var d = (r + 1) * sectors + s1;
                    tris.Add(a); tris.Add(c); tris.Add(b);
                    tris.Add(b); tris.Add(c); tris.Add(d);
                }
            }
            mesh.vertices = verts;
            mesh.SetTriangles(tris, 0);
        }

        private static void BuildSplit(Mesh mesh, Vector3[,] grid, int rings, int sectors)
        {
            var verts = new System.Collections.Generic.List<Vector3>(rings * sectors * 6);
            var tris = new System.Collections.Generic.List<int>(rings * sectors * 6);

            void Tri(Vector3 a, Vector3 b, Vector3 c)
            {
                tris.Add(verts.Count); verts.Add(a);
                tris.Add(verts.Count); verts.Add(b);
                tris.Add(verts.Count); verts.Add(c);
            }

            for (var r = 0; r < rings; r++)
            {
                for (var s = 0; s < sectors; s++)
                {
                    var s1 = (s + 1) % sectors;
                    Tri(grid[r, s], grid[r + 1, s], grid[r, s1]);
                    Tri(grid[r, s1], grid[r + 1, s], grid[r + 1, s1]);
                }
            }
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
        }
    }
}
