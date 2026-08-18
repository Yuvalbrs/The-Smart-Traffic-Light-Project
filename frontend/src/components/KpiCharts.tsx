import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DashboardFrame } from "../lib/wire";

interface Props {
  frames: DashboardFrame[];
}

const formatSimTime = (label: unknown): string => "t=" + Number(label).toFixed(0) + "s";

/**
 * Live KPI history (wait/queue/throughput), one point per real 1 Hz frame. Deliberately plots
 * only real frames, not the interpolated ticker values - a time series of interpolated points
 * would just redraw the same line at higher resolution.
 */
export function KpiCharts({ frames }: Props) {
  const rows = frames.map((f) => ({
    sim_time: f.sim_time,
    avg_wait_so_far: f.running_kpis.avg_wait_so_far,
    throughput_so_far: f.running_kpis.throughput_so_far,
    current_queue_total: f.running_kpis.current_queue_total,
  }));

  return (
    <div className="panel kpi-charts">
      <h2>Running KPIs (estimates - not the confirmatory results)</h2>
      <div className="kpi-chart-row">
        <div className="kpi-chart">
          <h3>avg wait so far (s)</h3>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="sim_time" fontSize={11} tickFormatter={(v: number) => v.toFixed(0)} />
              <YAxis fontSize={11} />
              <Tooltip labelFormatter={formatSimTime} />
              <Line type="monotone" dataKey="avg_wait_so_far" stroke="var(--chart-wait)" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="kpi-chart">
          <h3>throughput so far (veh/h)</h3>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="sim_time" fontSize={11} tickFormatter={(v: number) => v.toFixed(0)} />
              <YAxis fontSize={11} />
              <Tooltip labelFormatter={formatSimTime} />
              <Line type="monotone" dataKey="throughput_so_far" stroke="var(--chart-throughput)" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="kpi-chart">
          <h3>total queue (veh)</h3>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="sim_time" fontSize={11} tickFormatter={(v: number) => v.toFixed(0)} />
              <YAxis fontSize={11} />
              <Tooltip labelFormatter={formatSimTime} />
              <Line type="monotone" dataKey="current_queue_total" stroke="var(--chart-queue)" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
