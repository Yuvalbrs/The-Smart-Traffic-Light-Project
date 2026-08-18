import { useState } from "react";
import "./App.css";
import { ControlPanel } from "./components/ControlPanel";
import { ForecastPanel } from "./components/ForecastPanel";
import { KpiCharts } from "./components/KpiCharts";
import { MovementBars } from "./components/MovementBars";
import { PhaseIndicator } from "./components/PhaseIndicator";
import { ReplayBrowser } from "./components/ReplayBrowser";
import { useDashboardSocket } from "./lib/useDashboardSocket";
import type { SessionStatus } from "./lib/wire";

function App() {
  const [session, setSession] = useState<SessionStatus | null>(null);
  const { state, frames, latest, interpolated } = useDashboardSocket(true);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Smart Traffic Intersection — Live Dashboard</h1>
        <span className={"ws-badge ws-" + state}>ws/dashboard: {state}</span>
      </header>

      <main className="app-grid">
        <ControlPanel onSessionChange={setSession} />
        <PhaseIndicator
          currentPhase={latest?.current_phase ?? null}
          lastAction={latest?.last_action ?? null}
          simTime={session?.sim_time ?? latest?.sim_time ?? null}
        />
        <MovementBars
          queueLengths={interpolated?.queue_lengths ?? null}
          pressures={interpolated?.pressures ?? null}
        />
        <ForecastPanel forecast={latest?.forecast_next_30s ?? null} />
        <KpiCharts frames={frames} />
        <ReplayBrowser />
      </main>
    </div>
  );
}

export default App;
