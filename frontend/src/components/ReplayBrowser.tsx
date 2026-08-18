import { useEffect, useState } from "react";
import { ApiError, getRunKpis, getRunMetadata, listRuns } from "../lib/api";
import type { RunKpisResponse, RunMetadata, RunSummary } from "../lib/wire";

/**
 * T-05-02 scope: browse finished runs and inspect metadata/KPIs. Trace playback through the
 * live charts (and the Unity side) is T-05-04, not built here.
 */
export function ReplayBrowser() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [kpis, setKpis] = useState<RunKpisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = () => {
    listRuns(50, 0)
      .then((res) => {
        setRuns(res.runs);
        setTotal(res.total);
      })
      .catch((e) => setError(e instanceof ApiError ? e.detail : String(e)));
  };

  useEffect(refresh, []);

  const select = async (runId: string) => {
    setSelected(runId);
    setLoading(true);
    setError(null);
    try {
      const [meta, k] = await Promise.all([getRunMetadata(runId), getRunKpis(runId)]);
      setMetadata(meta);
      setKpis(k);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
      setMetadata(null);
      setKpis(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel replay-browser">
      <h2>Replay browser</h2>
      <div className="replay-layout">
        <div className="replay-list">
          <p className="control-note">
            {total} run{total === 1 ? "" : "s"} recorded
          </p>
          <button onClick={refresh} className="refresh-button">
            refresh
          </button>
          <ul>
            {runs.map((r) => (
              <li key={r.run_id}>
                <button
                  className={"run-row" + (r.run_id === selected ? " selected" : "")}
                  onClick={() => select(r.run_id)}
                >
                  <span className="run-name">{r.name ?? r.run_id}</span>
                  <span className="run-meta">
                    {r.controller ?? "?"} · {r.mode ?? "?"} · {r.created_at ?? "no timestamp"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="replay-detail">
          {loading && <p>loading…</p>}
          {error && <p className="error-text">{error}</p>}
          {metadata && (
            <>
              <h3>{metadata.name ?? metadata.run_id}</h3>
              <dl className="session-status">
                <dt>run_id</dt>
                <dd>{metadata.run_id}</dd>
                <dt>controller</dt>
                <dd>{metadata.controller}</dd>
                <dt>scenarios / seeds</dt>
                <dd>
                  {metadata.scenarios.join(", ") || "—"} / {metadata.seeds.join(", ") || "—"}
                </dd>
                <dt>episodes</dt>
                <dd>{metadata.episode_count}</dd>
                <dt>version chain</dt>
                <dd>
                  data={metadata.version_chain.data_version ?? "—"} · lstm=
                  {metadata.version_chain.lstm_version ?? "—"} · git=
                  {metadata.version_chain.git_sha ?? "—"} · sumo=
                  {metadata.version_chain.sumo_version ?? "—"}
                </dd>
              </dl>

              {kpis && kpis.rows.length > 0 && (
                <table className="kpi-table">
                  <thead>
                    <tr>
                      {kpis.columns.map((c) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {kpis.rows.map((row) => (
                      <tr key={row.episode_id} className={row.gridlock_censored ? "censored-row" : ""}>
                        {kpis.columns.map((c) => (
                          <td key={c}>
                            {(() => {
                              const v = (row as unknown as Record<string, unknown>)[c];
                              if (typeof v === "number") return v.toFixed(2);
                              if (typeof v === "boolean") return v ? "yes" : "no";
                              return v == null ? "—" : String(v);
                            })()}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {kpis && kpis.rows.length === 0 && <p className="control-note">no KPI rows for this run (live/incomplete session).</p>}
            </>
          )}
          {!metadata && !loading && !error && <p className="control-note">select a run to see its metadata and KPIs</p>}
        </div>
      </div>
    </div>
  );
}
