// Batch-mode build entry point for the Windows standalone viewer.
//
// The viewer has no authored scene: Bootstrap installs everything with
// [RuntimeInitializeOnLoadMethod(AfterSceneLoad)] and IntersectionScene builds the roads, the
// twelve signal heads and the camera at runtime. That is fine in the Editor, where Play mode
// always has a scene open, but a player build needs at least one scene in the build list or it
// ships with nothing to initialise into. So this creates a single empty scene, saves it, and
// builds with exactly that - the runtime hook does the rest, unchanged.
//
// Run it from the command line (the Editor must NOT have the project open - Unity 6 holds a
// lock on the project folder and the build aborts with "another Unity instance is running"):
//
//   Unity.exe -quit -batchmode -nographics \
//     -projectPath unity/SmartTrafficViz \
//     -executeMethod SmartTrafficViz.EditorTools.BuildScript.BuildWindows \
//     -logFile unity/build_standalone.log

using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SmartTrafficViz.EditorTools
{
    public static class BuildScript
    {
        private const string SceneDir = "Assets/Scenes";
        private const string ScenePath = SceneDir + "/Main.unity";

        /// <summary>Ensure a single empty scene exists and is the one the player boots into.</summary>
        private static string EnsureBootScene()
        {
            if (File.Exists(ScenePath))
            {
                return ScenePath;
            }

            if (!Directory.Exists(SceneDir))
            {
                Directory.CreateDirectory(SceneDir);
            }

            // Empty on purpose. Anything placed here would be a second source of truth for a
            // scene that IntersectionScene already builds at runtime, and the two would drift.
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.Refresh();
            Debug.Log($"[build] created boot scene {ScenePath}");
            return ScenePath;
        }

        public static void BuildWindows()
        {
            var scenePath = EnsureBootScene();
            var projectRoot = Directory.GetParent(Application.dataPath)!.FullName;
            var outputDir = Path.Combine(projectRoot, "Build");
            var outputExe = Path.Combine(outputDir, "SmartTrafficViz.exe");
            Directory.CreateDirectory(outputDir);

            var options = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = outputExe,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };

            // Windowed, not fullscreen: the point of this build is to sit BESIDE the dashboard,
            // and a viewer that grabs the whole screen on launch hides the numbers it explains.
            PlayerSettings.defaultIsNativeResolution = false;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.runInBackground = true;  // keep consuming the 1 Hz feed when unfocused
            PlayerSettings.productName = "Smart Traffic Viz";

            var report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;
            Debug.Log($"[build] result={summary.result} size={summary.totalSize} errors={summary.totalErrors}");

            if (summary.result != BuildResult.Succeeded)
            {
                // Exit non-zero so the caller can tell a failed build from a successful one
                // without parsing the log; batchmode otherwise reports success regardless.
                EditorApplication.Exit(1);
            }

            Debug.Log($"[build] OK -> {outputExe}");
            EditorApplication.Exit(0);
        }
    }
}
