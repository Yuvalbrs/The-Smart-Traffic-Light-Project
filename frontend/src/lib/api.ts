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
