import { useState } from "react";
import "./App.css";
import { ComparisonView } from "./components/ComparisonView";
import { ControlPanel } from "./components/ControlPanel";
import { ForecastPanel } from "./components/ForecastPanel";
import { KpiCharts } from "./components/KpiCharts";
import { MovementBars } from "./components/MovementBars";
import { PhaseIndicator } from "./components/PhaseIndicator";
import { ReplayBrowser } from "./components/ReplayBrowser";
import { TrainingPanel } from "./components/TrainingPanel";
import { useDashboardSocket } from "./lib/useDashboardSocket";
import type { SessionStatus } from "./lib/wire";

type Tab = "live" | "train" | "compare";

function App() {
  const [session, setSession] = useState<SessionStatus | null>(null);
  const [tab, setTab] = useState<Tab>("live");
  const { state, frames, latest, interpolated } = useDashboardSocket(true);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Smart Traffic Intersection — Live Dashboard</h1>
        <span className={"ws-badge ws-" + state}>ws/dashboard: {state}</span>
      </header>

      <nav className="tab-bar">
        <button className={"tab-button" + (tab === "live" ? " active" : "")} onClick={() => setTab("live")}>
          Live
        </button>
        <button className={"tab-button" + (tab === "train" ? " active" : "")} onClick={() => setTab("train")}>
          Train
        </button>
        <button className={"tab-button" + (tab === "compare" ? " active" : "")} onClick={() => setTab("compare")}>
          Compare
        </button>
      </nav>

      {tab === "live" && (
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
      )}

      {tab === "train" && (
        <main className="app-grid">
          <TrainingPanel />
        </main>
      )}

      {tab === "compare" && (
        <main className="app-grid">
          <ComparisonView />
        </main>
      )}
    </div>
  );
}

export default App;
