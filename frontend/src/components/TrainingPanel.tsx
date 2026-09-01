import { Fragment, useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  ApiError,
  WS_BASE,
  getControllers,
  getCurrentEvaluation,
  getCurrentTraining,
  getModels,
  startEvaluation,
  startTraining,
  stopEvaluation,
  stopTraining,
} from "../lib/api";
import type {
  EvaluationFrameMessage,
  EvaluationStatus,
  ModelInfo,
  TrainingFrameMessage,
  TrainingStatus,
  TrainingVariant,
} from "../lib/api";

const POLL_MS = 2000;
const MAX_EPISODES = 300;
// Used only until GET /controllers answers; the hub owns the real list.
const FALLBACK_EVAL_SCENARIOS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"];

/** The scenario whose demand comes from real measured counts in the database, not a formula. */
const MEASURED_SCENARIO = "SCN-R1";

type Demand = "synthetic" | "measured";
const MAX_EVAL_SEEDS = 15;

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
  // A 404 means "no job yet"; anything else means the hub is unreachable or broken. Swallowing
  // both rendered them identically as "no training job started yet", which is a lie when the
  // server is down - the one state a demo most needs to see.
  const [hubError, setHubError] = useState<string | null>(null);
  const [evalScenarios, setEvalScenarios] = useState<string[]>(FALLBACK_EVAL_SCENARIOS);
  const [demand, setDemand] = useState<Demand>("synthetic");

  // Mutated only from effect/socket callbacks (never during render) so the poll fallback below
  // can check "is the socket open right now" without forcing a re-render on every ws state flip.
  const wsStateRef = useRef<SocketState>("connecting");

  // --- trained models list (GET /models) --------------------------------
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // --- evaluation (POST/DELETE /evaluation) ------------------------------
  const [evalOpenFor, setEvalOpenFor] = useState<string | null>(null);
  const [evalScenario, setEvalScenario] = useState("SCN-05");
  const [evalSeeds, setEvalSeeds] = useState(5);
  const [evalStatus, setEvalStatus] = useState<EvaluationStatus | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalStopping, setEvalStopping] = useState(false);
  const evalWsStateRef = useRef<SocketState>("connecting");

  // 404 is the expected "nothing running" answer and clears the banner; every other failure is a
  // real one and must be shown rather than swallowed.
  const noteHubError = (e: unknown) => {
    if (e instanceof ApiError && e.status === 404) {
      setHubError(null);
      return;
    }
    setHubError(e instanceof ApiError ? e.detail : String(e));
  };

  useEffect(() => {
    let cancelled = false;
    getControllers()
      .then((opts) => {
        if (!cancelled && opts.scenarios.length > 0) setEvalScenarios(opts.scenarios);
      })
      .catch(() => {
        // the hub banner below already reports an unreachable server
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load whatever training job already exists (if any) and keep a live socket open for updates.
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    getCurrentTraining()
      .then((s) => {
        if (!cancelled) {
          setStatus(s);
          setHubError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) noteHubError(e);
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
        .then((s) => {
          setStatus(s);
          setHubError(null);
        })
        .catch(noteHubError);
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const fetchModels = () => {
    setModelsLoading(true);
    setModelsError(null);
    getModels()
      .then((res) => setModels(res.models))
      .catch((e) => setModelsError(e instanceof ApiError ? e.detail : String(e)))
      .finally(() => setModelsLoading(false));
  };

  // Load the model list once on mount.
  useEffect(() => {
    fetchModels();
  }, []);

  // A finished training job produces a new checkpoint that only shows up in GET /models once it
  // lands - re-list so the row (and its Evaluate button) appears without a manual refresh.
  useEffect(() => {
    if (status?.status === "done") fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.status, status?.job_id]);

  // Load whatever evaluation job already exists and keep a live socket open for updates - same
  // shape as the training socket above, just pointed at /ws/evaluation.
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    getCurrentEvaluation()
      .then((s) => {
        if (!cancelled) {
          setEvalStatus(s);
          setHubError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) noteHubError(e);
      });

    const connect = () => {
      if (cancelled) return;
      evalWsStateRef.current = "connecting";
      socket = new WebSocket(WS_BASE + "/ws/evaluation");

      socket.onopen = () => {
        evalWsStateRef.current = "open";
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data) as EvaluationFrameMessage;
        if (msg.type !== "evaluation_frame") return;
        if (!cancelled) setEvalStatus(msg);
      };

      socket.onclose = () => {
        if (cancelled) return;
        evalWsStateRef.current = "closed";
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

  // Fallback poll whenever the evaluation socket isn't open right now.
  useEffect(() => {
    const id = setInterval(() => {
      if (evalWsStateRef.current === "open") return;
      getCurrentEvaluation()
        .then((s) => setEvalStatus(s))
        .catch(() => {
          // nothing running yet, or a transient error - keep the last known status displayed
        });
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);

  // A finished evaluation is what the Compare tab reads - re-list models alongside it so
  // has_final / labels stay current too.
  useEffect(() => {
    if (evalStatus?.status === "done") fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalStatus?.status, evalStatus?.job_id]);

  const running = status?.status === "running";
  const evaluating = evalStatus?.status === "running";

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
        train_scenarios: demand === "measured" ? [MEASURED_SCENARIO] : null,
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

  const handleToggleEvalForm = (modelId: string) => {
    setEvalError(null);
    setEvalOpenFor((current) => (current === modelId ? null : modelId));
    setEvalScenario("SCN-05");
    setEvalSeeds(5);
  };

  const handleStartEvaluation = async (modelId: string) => {
    setEvalBusy(true);
    setEvalError(null);
    setEvalStopping(false);
    try {
      const s = await startEvaluation({ model_id: modelId, scenario: evalScenario, seeds: evalSeeds });
      setEvalStatus(s);
      setEvalOpenFor(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setEvalError("an evaluation is already running - stop it first");
      } else {
        setEvalError(e instanceof ApiError ? e.detail : String(e));
      }
    } finally {
      setEvalBusy(false);
    }
  };

  const handleStopEvaluation = async () => {
    setEvalBusy(true);
    setEvalError(null);
    try {
      const result = await stopEvaluation();
      setEvalStopping(result === "stopping");
    } catch (e) {
      setEvalError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setEvalBusy(false);
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
          training demand
          <select
            value={demand}
            onChange={(e) => setDemand(e.target.value as Demand)}
            disabled={running}
          >
            <option value="synthetic">synthetic scenarios (SCN-01/02/03)</option>
            <option value="measured">real measured - Hangzhou ({MEASURED_SCENARIO})</option>
          </select>
        </label>
        <label>
          label (optional)
          <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} disabled={running} />
        </label>
        <p className="control-note">30 episodes is a quick demo run; 300 episodes is a full run (~13 min).</p>
        {demand === "measured" ? (
          <p className="control-note">
            The arrival pattern comes from real measured traffic counts stored in the database;
            the agent learns by controlling it. A demonstration, not part of the pre-registered
            campaign - {MEASURED_SCENARIO} is deliberately outside the confirmatory set.
          </p>
        ) : (
          <p className="control-note">
            Demand is generated from the scenario's formula. Switch to measured to train against
            real recorded traffic instead.
          </p>
        )}
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
            {/* Read off the job, not off the label - so "trained on real data" is a fact the
                hub reports rather than something a demonstrator typed into a free-text box. */}
            <dt>training demand</dt>
            <dd>
              {status.train_scenarios
                ? `measured - ${status.train_scenarios.join(", ")}`
                : "synthetic (default rotation)"}
            </dd>
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
      {hubError && <p className="error-text">cannot reach the hub - {hubError}</p>}
      {!status && !hubError && <p className="control-note">no training job started yet</p>}

      <div className="models-section">
        <h3>Trained models</h3>
        {modelsLoading && <p className="control-note">loading…</p>}
        {modelsError && <p className="error-text">{modelsError}</p>}
        {!modelsLoading && !modelsError && models.length === 0 && (
          <p className="control-note">no models trained yet</p>
        )}

        {models.length > 0 && (
          <table className="models-table">
            <thead>
              <tr>
                <th>model</th>
                <th className="numeric-cell">episodes</th>
                <th>source</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <Fragment key={m.id}>
                  <tr>
                    <td>{m.label}</td>
                    <td className="numeric-cell">
                      {m.episodes_trained}/{m.episodes}
                    </td>
                    <td>
                      <span className={"source-badge source-" + m.source}>
                        {m.source === "user" ? "yours" : "campaign"}
                      </span>
                    </td>
                    <td>
                      <button
                        onClick={() => handleToggleEvalForm(m.id)}
                        disabled={!m.has_final || evaluating}
                        title={
                          !m.has_final
                            ? "this run never reached its final episode - nothing to evaluate yet"
                            : undefined
                        }
                      >
                        evaluate
                      </button>
                    </td>
                  </tr>
                  {evalOpenFor === m.id && (
                    <tr className="eval-form-row">
                      <td colSpan={4}>
                        <div className="control-form inline-eval-form">
                          <label>
                            scenario
                            <select value={evalScenario} onChange={(e) => setEvalScenario(e.target.value)}>
                              {evalScenarios.map((s) => (
                                <option key={s} value={s}>
                                  {s}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            seeds
                            <input
                              type="number"
                              min={1}
                              max={MAX_EVAL_SEEDS}
                              value={evalSeeds}
                              onChange={(e) => setEvalSeeds(Number(e.target.value))}
                            />
                          </label>
                          <div className="control-buttons">
                            <button onClick={() => handleStartEvaluation(m.id)} disabled={evalBusy || evaluating}>
                              run evaluation
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}

        {evalError && <p className="error-text">{evalError}</p>}
        {evalStopping && evaluating && <p className="control-note">stopping…</p>}

        {evalStatus && (
          <div className="evaluation-status">
            <dl className="session-status">
              <dt>job_id</dt>
              <dd>{evalStatus.job_id}</dd>
              <dt>status</dt>
              <dd className={"state-" + evalStatus.status}>{evalStatus.status}</dd>
              <dt>model</dt>
              <dd>{evalStatus.label}</dd>
              <dt>scenario / seeds</dt>
              <dd>
                {evalStatus.scenario} / {evalStatus.seeds}
              </dd>
              {evalStatus.error && (
                <>
                  <dt>error</dt>
                  <dd className="error-text">{evalStatus.error}</dd>
                </>
              )}
            </dl>

            <div className="progress-row">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(evalStatus.pct, 100)}%` }} />
              </div>
              <span className="progress-label">
                {evalStatus.episodes_done} / {evalStatus.episodes_total} episodes ({evalStatus.pct.toFixed(0)}%)
              </span>
            </div>

            <div className="control-buttons">
              <button onClick={handleStopEvaluation} disabled={evalBusy || !evaluating}>
                stop
              </button>
            </div>

            {evalStatus.status === "done" && (
              <p className="control-note">Evaluated. Open the Compare tab to see it.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
