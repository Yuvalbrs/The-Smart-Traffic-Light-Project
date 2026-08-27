/**
 * Types for the Phase-2 API contract (src/api/wire.py, src/api/server.py, src/api/live.py,
 * src/api/replay.py). Kept in one file so a wire-format change surfaces as a single diff here
 * instead of scattered `any`s.
 */

export const SCHEMA_VERSION = "1.1.0";

// --- WebSocket: /ws/dashboard --------------------------------------------

export interface HelloMessage {
  schema_version: string;
  type: "hello";
  channel: "dash" | "unity";
}

export interface RunningKpis {
  avg_wait_so_far: number;
  throughput_so_far: number;
  current_queue_total: number;
}

/** One frame from src/api/wire.py:dashboard_frame(). Mirrors the dict keys exactly. */
export interface DashboardFrame {
  schema_version: string;
  type: "dashboard_frame";
  seq: number;
  episode_id: number;
  sim_time: number;
  current_phase: number;
  last_action: number;
  /** M0..M11 halting counts */
  queue_lengths: number[];
  /** M0..M11, unnormalized */
  pressures: number[];
  running_kpis: RunningKpis;
  /** 36 = 3 horizons x 12 movements; null for every non-hybrid controller */
  forecast_next_30s: number[] | null;
}

export type DashboardSocketMessage = HelloMessage | DashboardFrame;

// --- REST: /controllers, /sessions ----------------------------------------

export interface ControllersResponse {
  controllers: string[];
  scenarios: string[];
  default: { controller: string; scenario: string; seed: number };
  note: string;
}

export interface StartSessionBody {
  controller: string;
  scenario: string;
  seed: number;
  episode_length_s: number;
  trace: boolean;
  /** Simulated seconds per wall second. 0 = as fast as the machine allows, 1 = real time. */
  speed: number;
}

export type SessionState = "starting" | "running" | "finished" | "failed" | "stopped";

export interface SessionStatus {
  run_id: string;
  controller: string;
  scenario: string;
  seed: number;
  state: SessionState;
  sim_time: number;
  frames: number;
  speed: number;
  error: string | null;
  started_at: number;
  finished_at: number | null;
  trace_path: string | null;
  loop_lag_s: { last: number; max: number };
}

// --- REST: /runs replay browser (T-05-02 scope: list + metadata/KPIs only) -

export interface VersionChain {
  data_version: string | null;
  lstm_version: string | null;
  git_sha: string | null;
  sumo_version: string | null;
  schema_version: string | null;
}

export interface RunSummary {
  run_id: string;
  name: string | null;
  mode: string | null;
  controller: string | null;
  created_at: string | null;
  version_chain: VersionChain;
}

export interface RunsListResponse {
  total: number;
  limit: number;
  offset: number;
  runs: RunSummary[];
}

export interface KpiFields {
  avg_waiting_time: number | null;
  avg_queue_length: number | null;
  throughput: number | null;
  num_stops: number | null;
  wait_p95: number | null;
  fairness_std: number | null;
  worst_movement_max_wait: number | null;
}

export interface EpisodePayload {
  episode_id: number;
  index_in_run: number;
  scenario: string | null;
  seed: number | null;
  total_reward: number | null;
  done_reason: string | null;
  loaded_count: number | null;
  departed_count: number | null;
  arrived_count: number | null;
  insertion_backlog_fraction: number | null;
  gridlock_censored: boolean;
  kpis: KpiFields | null;
}

export interface RunMetadata extends RunSummary {
  config: unknown;
  episode_count: number;
  scenarios: string[];
  seeds: number[];
  episodes: EpisodePayload[];
}

export interface RunKpisRow {
  episode_id: number;
  scenario: string | null;
  seed: number | null;
  gridlock_censored: boolean;
  avg_waiting_time: number | null;
  avg_queue_length: number | null;
  throughput: number | null;
  num_stops: number | null;
  wait_p95: number | null;
  fairness_std: number | null;
  worst_movement_max_wait: number | null;
}

export interface RunKpisResponse {
  run_id: string;
  columns: string[];
  rows: RunKpisRow[];
}

export interface HealthResponse {
  status: string;
  schema_version: string;
  database: string | null;
  session: SessionStatus | null;
  channels: Record<
    string,
    { subscribers: number; published: number; dropped: number; queue_maxsize: number }
  >;
}

// --- Static movement/phase metadata (specs/movements.yaml, frozen) --------

/** M0..M11 in canonical order: approach (N,E,S,W) x turn (left, through, right). */
export const MOVEMENT_LABELS: readonly string[] = [
  "N left", "N through", "N right",
  "E left", "E through", "E right",
  "S left", "S through", "S right",
  "W left", "W through", "W right",
];

export interface PhaseInfo {
  group: "NS" | "EW";
  name: string;
  desc: string;
}

/** Discrete(8) NEMA dual-ring phases, per specs/movements.yaml. */
export const PHASES: readonly PhaseInfo[] = [
  { group: "NS", name: "NS through", desc: "N-through + S-through" },
  { group: "NS", name: "NS left", desc: "N-left + S-left" },
  { group: "NS", name: "N approach", desc: "N-left + N-through" },
  { group: "NS", name: "S approach", desc: "S-left + S-through" },
  { group: "EW", name: "EW through", desc: "E-through + W-through" },
  { group: "EW", name: "EW left", desc: "E-left + W-left" },
  { group: "EW", name: "E approach", desc: "E-left + E-through" },
  { group: "EW", name: "W approach", desc: "W-left + W-through" },
];
