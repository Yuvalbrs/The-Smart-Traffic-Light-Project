import { MOVEMENT_LABELS } from "../lib/wire";

interface Props {
  forecast: number[] | null;
}

// ADR-006: the forecaster's horizons moved to 60/90/120s; the wire field name
// (`forecast_next_30s`) predates that and is kept for backward compatibility - label by
// what it actually is, not by the stale field name.
const HORIZON_LABELS = ["+60s", "+90s", "+120s"];

/** The frozen LSTM's 36-dim forecast (3 horizons x 12 movements). Null for non-hybrid controllers. */
export function ForecastPanel({ forecast }: Props) {
  return (
    <div className="panel forecast-panel">
      <h2>LSTM forecast</h2>
      {forecast == null ? (
        <p className="phase-meta">not available for this controller (non-hybrid)</p>
      ) : (
        <table className="forecast-table">
          <thead>
            <tr>
              <th>movement</th>
              {HORIZON_LABELS.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOVEMENT_LABELS.map((label, m) => (
              <tr key={label}>
                <td>{label}</td>
                {HORIZON_LABELS.map((_, h) => (
                  <td key={h}>{forecast[h * 12 + m]?.toFixed(2) ?? "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
