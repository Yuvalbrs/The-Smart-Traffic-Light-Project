// T-05-03 - the ws/unity wire contract, mirrored from src/api/wire.py.
//
// This is the tracer's own sim_frame envelope, relayed byte for byte (see
// tests/test_api_live.py::test_unity_socket_relays_sim_frames_untouched), which is what lets a
// live episode and a replayed one drive an identical scene.
//
// Newtonsoft rather than JsonUtility: signal_colors is a movement-id -> colour dictionary, and
// JsonUtility cannot deserialise dictionaries.

using System.Collections.Generic;
using Newtonsoft.Json;

namespace SmartTraffic
{
    /// <summary>One 1 Hz frame: <c>{type, seq, sim_time, episode_id, payload:{signal, vehicles}}</c>.</summary>
    public class SimFrame
    {
        [JsonProperty("type")] public string Type;
        [JsonProperty("schema_version")] public string SchemaVersion;
        [JsonProperty("seq")] public long Seq;
        [JsonProperty("sim_time")] public double SimTime;
        [JsonProperty("episode_id")] public long EpisodeId;
        [JsonProperty("transition")] public bool Transition;
        [JsonProperty("payload")] public SimPayload Payload;
    }

    public class SimPayload
    {
        [JsonProperty("signal")] public SignalState Signal;
        [JsonProperty("vehicles")] public List<VehicleState> Vehicles;
    }

    public class SignalState
    {
        [JsonProperty("phase_index")] public int PhaseIndex;

        /// <summary>
        /// Seconds left in the phase. Baseline controllers do not maintain this - webster runs
        /// emit values like 86099.0 (a day minus the sim clock) - so it is carried for
        /// completeness but never rendered. Phase colour comes from <see cref="SignalColors"/>.
        /// </summary>
        [JsonProperty("phase_remaining_s")] public double PhaseRemainingS;

        /// <summary>Movement id ("M0".."M11") -> "red" | "yellow" | "green".</summary>
        [JsonProperty("signal_colors")] public Dictionary<string, string> SignalColors;

        /// <summary>Raw 16-character SUMO state string, kept for debugging against traci.</summary>
        [JsonProperty("sumo_state")] public string SumoState;
    }

    public class VehicleState
    {
        [JsonProperty("id")] public string Id;

        /// <summary>SUMO x (metres, east positive). Maps to Unity <c>x</c>.</summary>
        [JsonProperty("x")] public float X;

        /// <summary>SUMO y (metres, north positive). Maps to Unity <c>z</c> - SUMO is 2-D, Unity is y-up.</summary>
        [JsonProperty("y")] public float Y;

        /// <summary>
        /// SUMO heading in degrees: 0 = north, increasing clockwise. Unity's y-Euler uses the
        /// same convention (0 = +z, clockwise seen from above), so it maps across unchanged.
        /// </summary>
        [JsonProperty("angle")] public float Angle;

        [JsonProperty("speed")] public float Speed;
        [JsonProperty("lane")] public string Lane;
        [JsonProperty("movement_id")] public string MovementId;
        [JsonProperty("type")] public string Type;
    }
}
