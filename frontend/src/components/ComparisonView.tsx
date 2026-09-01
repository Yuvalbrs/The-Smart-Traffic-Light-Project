import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getComparison, getControllers } from "../lib/api";
import type { ComparisonKpi, ComparisonResponse, ComparisonRow } from "../lib/api";

// The hub owns the scenario list (GET /controllers). This used to be a hard-coded five, which
// silently hid SCN-06..10 and SCN-R1 even once a user model had been evaluated on them - one of
// three separate copies of this list that had already drifted apart.
const FALLBACK_SCENARIOS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"];

function formatKpi(key: string, value: number | null): string {
  if (value == null) return "—";
  return key === "throughput" ? value.toFixed(0) : value.toFixed(2);
}

/** Best value per KPI column, respecting lower_is_better. A column with no numeric rows has no best. */
/**
 * Best value per KPI, computed over the CAMPAIGN rows only.
 *
 * User-trained models are excluded from winning a column. They are evaluated on different code
 * and usually on far fewer seeds, so a model trained for thirty seconds can top a column against
 * a 900-episode campaign purely by variance - which reads as a result and is not one. They are
 * still shown, and still comparable by eye; they just cannot be crowned.
 */
function bestValues(rows: ComparisonRow[], kpis: ComparisonKpi[]): Record<string, number> {
  const best: Record<string, number> = {};
  const campaignRows = rows.filter((r) => !r.is_user_model);
  for (const kpi of kpis) {
    const values = campaignRows
      .map((r) => (r as unknown as Record<string, unknown>)[kpi.key])
      .filter((v): v is number => typeof v === "number");
    if (values.length === 0) continue;
    best[kpi.key] = kpi.lower_is_better ? Math.min(...values) : Math.max(...values);
  }
  return best;
}

/** Confirmatory KPI comparison across controllers for one scenario (GET /comparison). */
/** The two headline KPIs, each charted on its own axis (see the comment at the charts). */
const CHARTED_KPIS = [
  { key: "avg_waiting_time", title: "Avg wait (s) - lower is better", fill: "var(--chart-wait)" },
  { key: "throughput", title: "Throughput (veh/h) - higher is better", fill: "var(--chart-throughput)" },
] as const;

export function ComparisonView() {
  const [scenario, setScenario] = useState("SCN-05");
  const [scenarios, setScenarios] = useState<string[]>(FALLBACK_SCENARIOS);
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // If the hub is unreachable the fallback list still renders a usable picker.
  useEffect(() => {
    let cancelled = false;
    getControllers()
      .then((opts) => {
        if (!cancelled && opts.scenarios.length > 0) setScenarios(opts.scenarios);
      })
      .catch(() => {
        // the comparison fetch below surfaces the connection error already
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getComparison(scenario)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.detail : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [scenario]);

  const best = data ? bestValues(data.rows, data.kpis) : {};

  return (
    <div className="panel comparison-view">
      <h2>Controller comparison</h2>
      <label>
        scenario
        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {scenarios.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      {loading && <p>loading…</p>}
      {error && <p className="error-text">{error}</p>}

      {data && data.rows.length > 0 && (
        <>
          <table className="comparison-table">
            <thead>
              <tr>
                <th>controller</th>
                <th className="numeric-cell">episodes</th>
                <th className="numeric-cell">gridlocked</th>
                {data.kpis.map((k) => (
                  <th key={k.key} className="numeric-cell">
                    {k.label} {k.lower_is_better ? "↓" : "↑"}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.controller} className={row.is_ours ? "ours-row" : ""}>
                  <td>
                    {row.label}
                    {row.is_user_model && (
                      <span className="yours-chip" title="trained inside this app, not part of the pre-registered campaign">
                        yours
                      </span>
                    )}
                  </td>
                  <td className="numeric-cell">{row.n_episodes}</td>
                  {/* Shown beside the KPIs, not among them: a controller that gridlocks most of
                      its episodes can post a flattering wait by never clearing the queue, so the
                      rate is the context that stops the rest of the row being read at face value.
                      Deliberately not eligible for a "best" marker - it is not a KPI to win on. */}
                  <td
                    className={
                      "numeric-cell" +
                      (row.gridlock_rate != null && row.gridlock_rate >= 0.5 ? " gridlock-high" : "")
                    }
                  >
                    {row.gridlock_rate == null ? "-" : `${Math.round(row.gridlock_rate * 100)}%`}
                  </td>
                  {data.kpis.map((k) => {
                    const value = (row as unknown as Record<string, unknown>)[k.key];
                    const numeric = typeof value === "number" ? value : null;
                    const isBest = !row.is_user_model && numeric != null && numeric === best[k.key];
                    return (
                      <td key={k.key} className={"numeric-cell" + (isBest ? " best-cell" : "")}>
                        {formatKpi(k.key, numeric)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {data.note && <p className="control-note">{data.note}</p>}
          {data.provenance.user_model_shas.length > 0 && (
            <p className="control-note">
              This scenario includes user-trained models, evaluated on different code than the
              pre-registered campaign ({data.provenance.user_model_shas.join(", ")}) - shown for
              comparison only.
            </p>
          )}
          <p className="control-note ablation-caption">
            Pre-registered ablation (n=15, Holm-corrected): adding the LSTM forecast to the agent's
            state significantly DEGRADES performance — avg wait +0.62 s (p=0.004), P95 wait +2.0 s
            (p=0.041). A random-forecast control is also significantly better than the real
            forecast, so the loss is attributable to the forecast information itself, not to the
            extra input capacity. This is a pre-registered negative result, not a bug.
          </p>

          {/* One chart per KPI, each with its own axis. A single plot carrying both put
              seconds and vehicles-per-hour on axes three orders of magnitude apart, which
              rendered the wait bars as slivers and invited a comparison between two
              quantities that share no unit. */}
          <div className="comparison-charts">
            {CHARTED_KPIS.map((kpi) => (
              <div className="comparison-chart" key={kpi.key}>
                <h4>{kpi.title}</h4>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart
                    data={data.rows.map((r) => ({
                      controller: r.label,
                      value: (r as unknown as Record<string, unknown>)[kpi.key] as number | null,
                    }))}
                    margin={{ top: 8, right: 8, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                    <XAxis dataKey="controller" fontSize={10} interval={0} angle={-20} textAnchor="end" height={70} />
                    <YAxis fontSize={11} width={54} />
                    <Tooltip
                      formatter={(value) =>
                        formatKpi(kpi.key, typeof value === "number" ? value : null)
                      }
                    />
                    <Bar dataKey="value" name={kpi.title} fill={kpi.fill} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </>
      )}
      {data && data.rows.length === 0 && <p className="control-note">no comparison data for this scenario yet</p>}
    </div>
  );
}
