# CLAUDE.md — Permanent Project Rules

This file is read by Claude Code on every session. Keep it tight. Keep it durable. Do not add
task-specific or transient content here.

## What this project is

Smart Traffic Intersection Management System — a final-year CS capstone (solo). A DQN agent controls
NEMA phase selection at a single 4-way intersection in SUMO simulation. State is augmented with a
frozen LSTM forecast. Evaluated against three baselines (Webster, max-pressure, SUMO actuated).

## The spec lives in the Obsidian vault (authoritative)

**Always start by reading the relevant spec files for the task.** The spec is authoritative. The
code conforms to the spec, not the other way around. The spec is NOT copied into this repo (to
avoid drift) — it lives in the vault:

`C:\Year3\Obsidian\Yuval\30_Projects\smart-traffic-rl\`

- Master task order: `backlog.md`
- Locked decisions ledger: `decisions.md`
- ADRs: `notes/adr-003-rl-algorithm.md`, `notes/adr-004-hybrid-integration.md`
- System architecture: `notes/system-architecture-overview.md`
- KPIs / metric formulas: `notes/kpis.md`
- Evaluation methodology: `notes/evaluation-methodology.md`
- Risk register and cut-points: `notes/risks-and-mitigations.md`
- Glossary (frozen): `final-glossary.md`
- Movement/phase + data specs: `specs/movements.yaml`, `specs/data-schema.md`
- Execution-readiness research: `notes/research/*.md`
- Open items / pending decisions: `notes/open-items.md`

## Hard rules — do not violate without explicit user approval

1. **The algorithm is locked.** Plain DQN (Double DQN as optional ablation). 2-layer MLP, hidden
   128. Do not propose PPO, A2C, SAC, or any other algorithm.
2. **The state is locked.** 20-dim base (12 pressures + 8-dim phase one-hot), optionally augmented
   to 56-dim via concatenation with the LSTM forecast. Do not change the dims.
3. **The action space is locked.** `Discrete(8)` over NEMA dual-ring phase pairs.
4. **The reward is locked.** `r = -|intersection pressure| - 0.1 * 1[a != a_prev]`. Switch penalty
   default 0.1.
5. **The baselines are locked.** Webster, max-pressure, SUMO actuated. Three. Mandatory.
6. **The architecture is locked.** Python FastAPI hub, SUMO via TraCI, WebSocket **1 Hz data push**
   (both Unity + dashboard; clients interpolate client-side — corrected from "5/10 Hz" on 2026-06-20,
   see vault `notes/open-items.md` F10 + `notes/unity-sumo-integration.md`), REST for replay, no auth.
7. **The provenance chain is mandatory.** Every checkpoint filename embeds `lstm_version`. Every
   SQLite results row records `(data_version, lstm_version, run_id, git_sha)`.

If the user pushes back on a locked decision, ask for explicit confirmation and update the relevant
ADR before changing code.

## Coding conventions

- Python 3.11+
- Formatting: `black` (line length 100)
- Linting: `ruff`
- Tests: `pytest` in `tests/`
- Type hints on all public functions
- Docstrings on all modules and public functions, NumPy style
- No premature abstraction. Concrete code first; refactor when the third use case appears.

## Repository layout

See `README.md`. Authoritative naming: `src/{env,ml,baselines,data,api,metrics}/`, plus `scripts/`,
`config/`, `tests/`, `frontend/`, `unity/`, `docs/` (signpost to the vault). The `src/` naming is
canonical; the older `simulator/`/`agents/`/`forecaster/`/`backend/` names were reconciled out of the
vault on 2026-06-20.

## Workflow rules

- **One task at a time.** Reference the task ID from `backlog.md` (e.g., T-00-01). Finish it,
  commit, then move on.
- **Propose before coding.** For any task larger than a single small file, propose the file
  structure and high-level approach first. Wait for approval.
- **Reference the spec by file path.** When making a design choice, cite the spec file that
  justifies it. If the spec is silent, surface that gap to the user instead of inventing an answer.
- **Commit after every passing task.** Use the task ID in the commit message:
  `git commit -m "T-00-01: vertical slice passing"`.
- **Push back on scope creep.** If the user asks for something not in the current task's DoD, ask
  whether to expand the current task or queue a new one. Don't silently add scope.
- **Reproducibility is non-negotiable.** Same seed -> same outputs. Test this whenever introducing
  randomness.

## What to do if something is unclear

Stop and ask. Do not guess. The cost of an extra question is seconds; the cost of building the wrong
thing is hours.

## What to do if the spec is wrong

The spec can be wrong. ADRs can be amended. But amendments happen through an explicit conversation
with the user, who updates the ADR. Code that silently contradicts the spec is the failure mode to
avoid.
