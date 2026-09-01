import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, WS_BASE, getCurrentTraining, startTraining, stopTraining } from "../lib/api";
import type { TrainingFrameMessage, TrainingStatus, TrainingVariant } from "../lib/api";

const POLL_MS = 2000;
const MAX_EPISODES = 300;

type SocketState = "connecting" | "open" | "closed";

const VARIANTS: { value: TrainingVariant; label: string }[] = [
  { value: "plain", label: "plain DQN" },
  { value: "hybrid", label: "DQN + forecast (hybrid)" },
  { value: "random-lstm", label: "DQN + random LSTM (ablation)" },
];

const formatEpisode = (label: unknown): string => "episode " + label;

/**
 * Start/monitor a training job (POST/DELETE /training). Live updates arrive over /ws/training;
 * when that socket isn't open we fall back to polling GET /training/current so the panel is
 * never dead.
 */
export function TrainingPanel() {
  const [variant, setVariant] = useState<TrainingVariant>("hybrid");
  const [seed, setSeed] = useState(42);
  const [episodes, setEpisodes] = useState(30);
  const [label, setLabel] = useState("");
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  // Mutated only from effect/socket callbacks (never during render) so the poll fallback below
  // can check "is the socket open right now" without forcing a re-render on every ws state flip.
  const wsStateRef = useRef<SocketState>("connecting");

  // Load whatever training job already exists (if any) and keep a live socket open for updates.
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    getCurrentTraining()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        // 404: no training job has ever started - status stays null.
      });

    const connect = () => {
      if (cancelled) return;
      wsStateRef.current = "connecting";
      socket = new WebSocket(WS_BASE + "/ws/training");

      socket.onopen = () => {
        wsStateRef.current = "open";
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data) as TrainingFrameMessage;
        if (msg.type !== "training_frame") return;
        if (!cancelled) setStatus(msg);
      };

      socket.onclose = () => {
        if (cancelled) return;
        wsStateRef.current = "closed";
        retryTimer = setTimeout(connect, 1500);
      };

      socket.onerror = () => socket?.close();
    };
    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  // Fallback poll whenever the socket isn't open right now.
  useEffect(() => {
    const id = setInterval(() => {
      if (wsStateRef.current === "open") return;
      getCurrentTraining()
        .then((s) => setStatus(s))
        .catch(() => {
          // nothing running yet, or a transient error - keep the last known status displayed
        });
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const running = status?.status === "running";

  const handleStart = async () => {
    setBusy(true);
    setError(null);
    setStopping(false);
    try {
      const s = await startTraining({
        variant,
        seed,
        episodes,
        episode_length_s: null,
        label: label.trim() ? label.trim() : null,
      });
      setStatus(s);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError("a training job (or live session) is already running - stop it first");
      } else {
        setError(e instanceof ApiError ? e.detail : String(e));
      }
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await stopTraining();
      setStopping(result === "stopping");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel training-panel">
      <h2>Train a controller</h2>
      <div className="control-form">
        <label>
          variant
          <select value={variant} onChange={(e) => setVariant(e.target.value as TrainingVariant)} disabled={running}>
            {VARIANTS.map((v) => (
              <option key={v.value} value={v.value}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          seed
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} disabled={running} />
        </label>
        <label>
          episodes
          <input
            type="number"
            min={1}
            max={MAX_EPISODES}
            value={episodes}
            onChange={(e) => setEpisodes(Number(e.target.value))}
            disabled={running}
          />
        </label>
        <label>
          label (optional)
          <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} disabled={running} />
        </label>
        <p className="control-note">30 episodes is a quick demo run; 300 episodes is a full run (~13 min).</p>
        <div className="control-buttons">
          <button onClick={handleStart} disabled={busy || running}>
            start training
          </button>
          <button onClick={handleStop} disabled={busy || !running}>
            stop
          </button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}
      {stopping && running && <p className="control-note">stopping…</p>}

      {status && (
        <div className="training-status">
          <dl className="session-status">
            <dt>job_id</dt>
            <dd>{status.job_id}</dd>
            <dt>status</dt>
            <dd className={"state-" + status.status}>{status.status}</dd>
            <dt>variant / seed</dt>
            <dd>
              {status.variant} / {status.seed}
            </dd>
            {status.label && (
              <>
                <dt>label</dt>
                <dd>{status.label}</dd>
              </>
            )}
            <dt>run_dir</dt>
            <dd>{status.run_dir}</dd>
            {status.error && (
              <>
                <dt>error</dt>
                <dd className="error-text">{status.error}</dd>
              </>
            )}
          </dl>

          <div className="progress-row">
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.min(status.pct, 100)}%` }} />
            </div>
            <span className="progress-label">
              {status.episodes_done} / {status.episodes} episodes ({status.pct.toFixed(0)}%)
            </span>
          </div>

          {status.curve.length > 0 && (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={status.curve}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="ep" fontSize={11} />
                <YAxis yAxisId="reward" fontSize={11} />
                <YAxis yAxisId="epsilon" orientation="right" fontSize={11} domain={[0, 1]} />
                <Tooltip labelFormatter={formatEpisode} />
                <Line
                  yAxisId="reward"
                  type="monotone"
                  dataKey="reward"
                  name="reward"
                  stroke="var(--chart-throughput)"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="epsilon"
                  type="monotone"
                  dataKey="epsilon"
                  name="epsilon"
                  stroke="var(--chart-pressure)"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
      {!status && <p className="control-note">no training job started yet</p>}
    </div>
  );
}
