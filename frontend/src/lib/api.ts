/** REST client for the FastAPI hub (src/api/server.py, src/api/replay.py). No auth, single machine. */

import type {
  ControllersResponse,
  RunKpisResponse,
  RunMetadata,
  RunsListResponse,
  SessionStatus,
  StartSessionBody,
} from "./wire";

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
export const WS_BASE = API_BASE.replace(/^http/, "ws");

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // body wasn't JSON; keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const getControllers = () => request<ControllersResponse>("/controllers");

export const getCurrentSession = () => request<SessionStatus>("/sessions/current");

export const startSession = (body: StartSessionBody) =>
  request<SessionStatus>("/sessions", { method: "POST", body: JSON.stringify(body) });

export const stopSession = () => request<void>("/sessions/current", { method: "DELETE" });

export const listRuns = (limit = 50, offset = 0) =>
  request<RunsListResponse>(`/runs?limit=${limit}&offset=${offset}`);

export const getRunMetadata = (runId: string) =>
  request<RunMetadata>(`/runs/${encodeURIComponent(runId)}/metadata`);

export const getRunKpis = (runId: string) =>
  request<RunKpisResponse>(`/runs/${encodeURIComponent(runId)}/kpis`);

// --- REST: /models ---------------------------------------------------------

export interface ModelInfo {
  id: string;
  variant: string;
  seed: number;
  episodes: number;
  obs_dim: number;
  lstm_version: string | null;
  git_sha: string | null;
  has_final: boolean;
  label: string;
  source: "matrix" | "user";
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export const getModels = () => request<ModelsResponse>("/models");

// --- REST + WS: /training, /ws/training -------------------------------------

export type TrainingVariant = "plain" | "hybrid" | "random-lstm";
export type TrainingJobStatus = "running" | "done" | "failed" | "cancelled";

export interface StartTrainingBody {
  variant: TrainingVariant;
  seed: number;
  episodes: number;
  episode_length_s: number | null;
  label: string | null;
}

export interface TrainingCurvePoint {
  ep: number;
  reward: number;
  epsilon: number;
}

export interface TrainingStatus {
  job_id: string;
  status: TrainingJobStatus;
  variant: string;
  seed: number;
  episodes: number;
  label: string | null;
  episodes_done: number;
  pct: number;
  curve: TrainingCurvePoint[];
  run_dir: string;
  started_at: string;
  error: string | null;
}

/** WS /ws/training push - a TrainingStatus wrapped in the usual envelope fields. */
export interface TrainingFrameMessage extends TrainingStatus {
  schema_version: string;
  type: "training_frame";
}

export const getCurrentTraining = () => request<TrainingStatus>("/training/current");

export const startTraining = (body: StartTrainingBody) =>
  request<TrainingStatus>("/training", { method: "POST", body: JSON.stringify(body) });

export type StopTrainingResult = "stopped" | "stopping";

/**
 * DELETE /training/current distinguishes 204 (stopped) from 202 (still stopping) - the generic
 * request() helper only sees "ok", so this reimplements the fetch to read the status code.
 */
export async function stopTraining(): Promise<StopTrainingResult> {
  const res = await fetch(API_BASE + "/training/current", { method: "DELETE" });
  if (res.status === 204) return "stopped";
  if (res.status === 202) return "stopping";
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail ?? detail;
  } catch {
    // body wasn't JSON; keep statusText
  }
  throw new ApiError(res.status, detail);
}

// --- REST: /comparison -------------------------------------------------------

export interface ComparisonKpi {
  key: string;
  label: string;
  lower_is_better: boolean;
}

export interface ComparisonRow {
  controller: string;
  label: string;
  is_ours: boolean;
  n_episodes: number;
  avg_waiting_time: number | null;
  throughput: number | null;
  avg_queue_length: number | null;
  wait_p95: number | null;
  worst_movement_max_wait: number | null;
  gridlock_rate: number | null;
}

export interface ComparisonResponse {
  scenario: string;
  rows: ComparisonRow[];
  kpis: ComparisonKpi[];
  note: string;
}

export const getComparison = (scenario: string) =>
  request<ComparisonResponse>(`/comparison?scenario=${encodeURIComponent(scenario)}`);
