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
        /// <summary>Must match <c>IntersectionScene.PinnedMaterial</c>, which loads it by name.</summary>
        private const string PinnedMaterialName = "ScenePinMaterial";

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

        /// <summary>Create the Resources material that pins a real shader into the player build.
        ///
        /// This project authors no materials - every mesh is generated at runtime and painted with
        /// `new Material(Shader.Find(...))`. That works in the Editor, where every built-in shader
        /// is loaded, and FAILS in a build, which only ships shaders some asset references. The
        /// first standalone build proved it: `ArgumentNullException: Parameter name: shader` inside
        /// IntersectionScene.Paint, so the whole 3-D scene never built.
        ///
        /// One material asset under Resources/ is the fix: Resources content is always included,
        /// so the shader it points at is too. Created here rather than committed as hand-written
        /// YAML so the shader reference is resolved by Unity itself and cannot rot.</summary>
        private static void EnsureShaderPin()
        {
            const string dir = "Assets/Resources";
            const string path = dir + "/" + PinnedMaterialName + ".mat";

            if (File.Exists(path))
            {
                Debug.Log($"[build] shader pin already present: {path}");
                return;
            }
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            if (shader == null)
            {
                Debug.LogError("[build] no Lit/Standard shader in the Editor either - cannot pin one.");
                return;
            }

            AssetDatabase.CreateAsset(new Material(shader), path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[build] created shader pin {path} using '{shader.name}'");
        }

        public static void BuildWindows()
        {
            EnsureShaderPin();
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
