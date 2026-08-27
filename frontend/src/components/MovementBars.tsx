import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MOVEMENT_LABELS } from "../lib/wire";

interface Props {
  queueLengths: number[] | null;
  pressures: number[] | null;
}

/** Per-movement (M0-M11) queue length and pressure, fed from the interpolated live frame. */
export function MovementBars({ queueLengths, pressures }: Props) {
  const rows = MOVEMENT_LABELS.map((label, i) => ({
    movement: label,
    queue: queueLengths?.[i] ?? 0,
    pressure: pressures?.[i] ?? 0,
  }));

  return (
    <div className="panel movement-bars">
      <h2>Per-movement queues &amp; pressure</h2>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="movement" angle={-40} textAnchor="end" interval={0} height={60} fontSize={11} />
          <YAxis yAxisId="queue" fontSize={11} />
          <Tooltip />
          {/* isAnimationActive=false is load-bearing: this chart is fed the interpolated frame,
              which useDashboardSocket refreshes every requestAnimationFrame (~60 Hz). Recharts
              restarts a bar's enter animation on each data change, so an animated bar never gets
              past height 0 and renders as an empty <g>. */}
          <Bar
            yAxisId="queue"
            dataKey="queue"
            name="queue length"
            fill="var(--chart-queue)"
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="movement" angle={-40} textAnchor="end" interval={0} height={60} fontSize={11} />
          <YAxis yAxisId="pressure" fontSize={11} />
          <Tooltip />
          <Bar
            yAxisId="pressure"
            dataKey="pressure"
            name="pressure"
            fill="var(--chart-pressure)"
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
