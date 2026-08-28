// T-05-03 - zero-click start.
//
// The viewer installs itself after any scene loads, so the project needs no authored scene asset
// and no GameObject dragged into a hierarchy: open the project and press Play. That keeps the
// whole client reviewable as text in git - a .unity scene file is a binary-ish YAML blob that
// nobody can diff - and removes the Editor steps most likely to go wrong under deadline.

using UnityEngine;

namespace SmartTraffic
{
    public static class Bootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Install()
        {
            if (Object.FindFirstObjectByType<TrafficViz>() != null) return;
            var host = new GameObject("SmartTrafficViz");
            host.AddComponent<TrafficViz>();
            Object.DontDestroyOnLoad(host);
        }
    }
}
