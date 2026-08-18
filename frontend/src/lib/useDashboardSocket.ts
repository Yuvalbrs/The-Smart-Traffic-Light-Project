import { useEffect, useRef, useState } from "react";
import { WS_BASE } from "./api";
import type { DashboardFrame, DashboardSocketMessage } from "./wire";

const HISTORY_LIMIT = 180; // 3 minutes at 1 Hz - enough for the live charts without unbounded growth

export type SocketState = "connecting" | "open" | "closed";

export interface InterpolatedFrame {
  queue_lengths: number[];
  pressures: number[];
  avg_wait_so_far: number;
  throughput_so_far: number;
  current_queue_total: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpArray(a: number[], b: number[], t: number): number[] {
  if (a.length !== b.length) return b;
  return a.map((v, i) => lerp(v, b[i], t));
}

/**
 * Subscribes to ws/dashboard and reconnects on drop. Frames arrive at 1 Hz (locked
 * architecture, F10); `interpolated` ticks on every animation frame between the last two
 * real frames so numeric displays move smoothly instead of stepping once a second.
 */
export function useDashboardSocket(enabled: boolean) {
  const [state, setState] = useState<SocketState>("connecting");
  const [frames, setFrames] = useState<DashboardFrame[]>([]);
  const [interpolated, setInterpolated] = useState<InterpolatedFrame | null>(null);

  const prevRef = useRef<{ frame: DashboardFrame; at: number } | null>(null);
  const latestRef = useRef<{ frame: DashboardFrame; at: number } | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) {
      setState("closed");
      return;
    }

    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      setState("connecting");
      socket = new WebSocket(WS_BASE + "/ws/dashboard");

      socket.onopen = () => setState("open");

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data) as DashboardSocketMessage;
        if (msg.type !== "dashboard_frame") return;
        const now = performance.now();
        if (latestRef.current) prevRef.current = latestRef.current;
        latestRef.current = { frame: msg, at: now };
        setFrames((prior) => {
          const next = [...prior, msg];
          return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next;
        });
      };

      socket.onclose = () => {
        if (cancelled) return;
        setState("closed");
        retryTimer = setTimeout(connect, 1500);
      };

      socket.onerror = () => socket?.close();
    };

    connect();

    const tick = () => {
      const latest = latestRef.current;
      if (latest) {
        const prev = prevRef.current ?? latest;
        // Frames land ~1s apart; clamp so a stalled connection freezes on the last real value
        // instead of extrapolating forever.
        const t = Math.min((performance.now() - latest.at) / 1000, 1);
        setInterpolated({
          queue_lengths: lerpArray(prev.frame.queue_lengths, latest.frame.queue_lengths, t),
          pressures: lerpArray(prev.frame.pressures, latest.frame.pressures, t),
          avg_wait_so_far: lerp(
            prev.frame.running_kpis.avg_wait_so_far,
            latest.frame.running_kpis.avg_wait_so_far,
            t,
          ),
          throughput_so_far: lerp(
            prev.frame.running_kpis.throughput_so_far,
            latest.frame.running_kpis.throughput_so_far,
            t,
          ),
          current_queue_total: lerp(
            prev.frame.running_kpis.current_queue_total,
            latest.frame.running_kpis.current_queue_total,
            t,
          ),
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      socket?.close();
      prevRef.current = null;
      latestRef.current = null;
    };
  }, [enabled]);

  const latest = frames.length > 0 ? frames[frames.length - 1] : null;
  return { state, frames, latest, interpolated };
}
