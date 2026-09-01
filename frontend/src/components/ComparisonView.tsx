import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getComparison } from "../lib/api";
import type { ComparisonKpi, ComparisonResponse, ComparisonRow } from "../lib/api";

const SCENARIOS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"];

function formatKpi(key: string, value: number | null): string {
  if (value == null) return "—";
  return key === "throughput" ? value.toFixed(0) : value.toFixed(2);
}

/** Best value per KPI column, respecting lower_is_better. A column with no numeric rows has no best. */
function bestValues(rows: ComparisonRow[], kpis: ComparisonKpi[]): Record<string, number> {
  const best: Record<string, number> = {};
  for (const kpi of kpis) {
    const values = rows
      .map((r) => (r as unknown as Record<string, unknown>)[kpi.key])
      .filter((v): v is number => typeof v === "number");
    if (values.length === 0) continue;
    best[kpi.key] = kpi.lower_is_better ? Math.min(...values) : Math.max(...values);
  }
  return best;
}

/** Confirmatory KPI comparison across controllers for one scenario (GET /comparison). */
export function ComparisonView() {
  const [scenario, setScenario] = useState("SCN-05");
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          {SCENARIOS.map((s) => (
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
                  <td>{row.label}</td>
                  <td className="numeric-cell">{row.n_episodes}</td>
                  {data.kpis.map((k) => {
                    const value = (row as unknown as Record<string, unknown>)[k.key];
                    const numeric = typeof value === "number" ? value : null;
                    const isBest = numeric != null && numeric === best[k.key];
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

          <ResponsiveContainer width="100%" height={240}>
            <BarChart
              data={data.rows.map((r) => ({
                controller: r.label,
                avg_waiting_time: r.avg_waiting_time,
                throughput: r.throughput,
              }))}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="controller" fontSize={11} />
              <YAxis yAxisId="wait" fontSize={11} />
              <YAxis yAxisId="throughput" orientation="right" fontSize={11} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="wait" dataKey="avg_waiting_time" name="avg wait (s)" fill="var(--chart-wait)" />
              <Bar yAxisId="throughput" dataKey="throughput" name="throughput (veh/h)" fill="var(--chart-throughput)" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
      {data && data.rows.length === 0 && <p className="control-note">no comparison data for this scenario yet</p>}
    </div>
  );
}
