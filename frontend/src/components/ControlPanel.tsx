import { useEffect, useState } from "react";
import {
  ApiError,
  getControllers,
  getCurrentSession,
  getViewer,
  startSession,
  startViewer,
  stopSession,
} from "../lib/api";
import type { ViewerStatus } from "../lib/api";
import type { ControllersResponse, SessionStatus } from "../lib/wire";

const POLL_MS = 2000;

/** Controller/scenario/seed picker + session lifecycle (POST/DELETE /sessions). */
export function ControlPanel({ onSessionChange }: { onSessionChange: (s: SessionStatus | null) => void }) {
  const [options, setOptions] = useState<ControllersResponse | null>(null);
  const [controller, setController] = useState("");
  const [scenario, setScenario] = useState("");
  const [seed, setSeed] = useState(7000);
  const [episodeLengthS, setEpisodeLengthS] = useState(600);
  const [trace, setTrace] = useState(true);
  // Default to real time: unpaced, libsumo steps ~700x faster than the wall clock and the charts
  // land fully drawn. Demos want to be watchable; set 0 to go as fast as the machine allows.
  const [speed, setSpeed] = useState(1);
  const [session, setSession] = useState<SessionStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A rejected /controllers call used to leave `options` null forever, and the whole form sits
  // behind `{options && ...}` - so with the hub down this panel rendered blank and said nothing.
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  // The hub can launch the built Unity player (POST /viewer). The endpoints existed but nothing
  // called them, so opening the 3-D view meant finding an .exe by hand or opening the editor -
  // three manual steps in a demo that is otherwise one double-click.
  const [viewer, setViewer] = useState<ViewerStatus | null>(null);
  const [viewerBusy, setViewerBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setOptionsError(null);
    getControllers()
      .then((opts) => {
        if (cancelled) return;
        setOptions(opts);
        setController(opts.default.controller);
        setScenario(opts.default.scenario);
        setSeed(opts.default.seed);
      })
      .catch((e) => {
        if (!cancelled) setOptionsError(e instanceof ApiError ? e.detail : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await getCurrentSession();
        if (!cancelled) {
          setSession(s);
          onSessionChange(s);
        }
      } catch (e) {
        if (!cancelled && !(e instanceof ApiError && e.status === 404)) {
          // transient network error while polling - leave the last known status displayed
        }
      }
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    getViewer()
      .then((v) => {
        if (!cancelled) setViewer(v);
      })
      .catch(() => {
        // the hub banner above already reports an unreachable server
      });
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  const handleOpenViewer = async () => {
    setViewerBusy(true);
    setError(null);
    try {
      setViewer(await startViewer());
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setViewerBusy(false);
    }
  };

  const running = session?.state === "starting" || session?.state === "running";

  const handleStart = async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await startSession({
        controller,
        scenario,
        seed,
        episode_length_s: episodeLengthS,
        trace,
        speed,
      });
      setSession(s);
      onSessionChange(s);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    setError(null);
    try {
      await stopSession();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel control-panel">
      <h2>Session control</h2>
      {!options && optionsError && (
        <div className="control-offline">
          <p className="error-text">cannot reach the hub - {optionsError}</p>
          <button onClick={() => setReloadKey((k) => k + 1)}>retry</button>
        </div>
      )}
      {!options && !optionsError && <p className="control-note">loading options...</p>}
      {options && (
        <div className="control-form">
          <label>
            controller
            <select value={controller} onChange={(e) => setController(e.target.value)} disabled={running}>
              {options.controllers.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            scenario
            <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={running}>
              {options.scenarios.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label>
            seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              disabled={running}
            />
          </label>
          <label>
            episode length (s)
            <input
              type="number"
              min={60}
              max={3600}
              value={episodeLengthS}
              onChange={(e) => setEpisodeLengthS(Number(e.target.value))}
              disabled={running}
            />
          </label>
          <label>
            speed (sim s per real s)
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={running}
            >
              <option value={1}>1x - real time (watchable)</option>
              <option value={5}>5x</option>
              <option value={10}>10x</option>
              <option value={0}>unpaced - as fast as possible</option>
            </select>
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={trace} onChange={(e) => setTrace(e.target.checked)} disabled={running} />
            record trace + provenance row
          </label>
          <p className="control-note">{options.note}</p>
          <div className="control-buttons">
            <button onClick={handleStart} disabled={busy || running}>
              start session
            </button>
            <button onClick={handleStop} disabled={busy || !running}>
              stop session
            </button>
          </div>
          {viewer && (
            <div className="viewer-launch">
              <button
                onClick={handleOpenViewer}
                disabled={viewerBusy || !viewer.available || viewer.running}
                title={viewer.available ? viewer.path : (viewer.hint ?? "")}
              >
                {viewer.running ? "3-D view is open" : "open 3-D view"}
              </button>
              <p className="control-note">
                {viewer.available
                  ? "Opens the Unity window on this same episode feed."
                  : viewer.hint}
              </p>
            </div>
          )}
        </div>
      )}
      {error && <p className="error-text">{error}</p>}
      {session && (
        <dl className="session-status">
          <dt>run_id</dt>
          <dd>{session.run_id}</dd>
          <dt>state</dt>
          <dd className={"state-" + session.state}>{session.state}</dd>
          <dt>controller / scenario / seed</dt>
          <dd>
            {session.controller} / {session.scenario} / {session.seed}
          </dd>
          <dt>frames</dt>
          <dd>{session.frames}</dd>
          <dt>speed</dt>
          <dd>{session.speed > 0 ? `${session.speed}x real time` : "unpaced"}</dd>
          <dt>loop lag (last / max)</dt>
          <dd>
            {session.loop_lag_s.last.toFixed(3)}s / {session.loop_lag_s.max.toFixed(3)}s
          </dd>
          {session.error && (
            <>
              <dt>error</dt>
              <dd className="error-text">{session.error}</dd>
            </>
          )}
        </dl>
      )}
    </div>
  );
}
