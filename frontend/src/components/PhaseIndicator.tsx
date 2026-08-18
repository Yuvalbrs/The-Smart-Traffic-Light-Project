import { PHASES } from "../lib/wire";

interface Props {
  currentPhase: number | null;
  lastAction: number | null;
  simTime: number | null;
}

/** Current NEMA phase + whether the lights are mid-transition (last_action != current_phase). */
export function PhaseIndicator({ currentPhase, lastAction, simTime }: Props) {
  const phase = currentPhase != null ? PHASES[currentPhase] : null;
  const transitioning = currentPhase != null && lastAction != null && currentPhase !== lastAction;

  return (
    <div className="panel phase-indicator">
      <h2>Signal phase</h2>
      {phase ? (
        <>
          <div className={"phase-badge phase-group-" + phase.group.toLowerCase()}>
            <span className="phase-number">P{currentPhase}</span>
            <span className="phase-name">{phase.name}</span>
          </div>
          <p className="phase-desc">{phase.desc}</p>
          <p className="phase-meta">
            group {phase.group}
            {transitioning && <span className="transitioning"> · transitioning (yellow/all-red)</span>}
          </p>
          <p className="phase-meta">sim time: {simTime != null ? simTime.toFixed(1) + " s" : "—"}</p>
        </>
      ) : (
        <p className="phase-meta">no live session</p>
      )}
    </div>
  );
}
