export const meta = {
  name: 'readiness-audit',
  description:
    'Multi-layer production-readiness audit of smart-traffic-rl: one reviewer per layer, ' +
    'adversarial verification of every finding, then one ranked report.',
}

// Codifies the review that found the anti-starvation bound, the survivorship-biased KPIs,
// the fabricated statistical pairs and the non-null control on 2026-08-28. Re-run it after
// any substantial change to the env, the training stack or the analysis.
//
// Usage:  /readiness-audit
//         Run /readiness-audit on src/env and src/ml     (args narrows the layers)

const REPO = 'C:/Year3/smart-traffic-rl'
const VAULT = 'C:/Year3/Obsidian/Yuval/30_Projects/smart-traffic-rl'

// The defect history every reviewer needs, so nobody re-reports known-fixed issues and
// everybody knows which failure CLASS actually slips past this project's test suite.
const CONTEXT = `
You are auditing a SUMO + Deep-RL traffic-signal capstone at ${REPO}.
The authoritative spec lives in an Obsidian vault at ${VAULT} (specs/movements.yaml is the
movement/phase SSOT; preregistration.md is a FROZEN statistical contract; decisions.md holds
the defect history). READ-ONLY: do not edit any file.

Six defects have already been found and FIXED - do not re-report them, but do check they are
still fixed and look for anything of the same CLASS:
1. green_state emitted 'G' (green-major) for the four free right turns; they must be 'g'.
2. The link binding gave the shared right+through lane's THROUGH links to the free right
   movement, so through traffic was green in every phase.
3. SUMO's default jmTimegapMinor=1.0s let yielding turns dart into traffic.
4. netconvert bakes the junction right-of-way 'response' matrix from the TLS program it is
   given at BUILD time; no program was supplied, so runtime 'g' could not make links yield.
5. max_red_s (anti-starvation) was specified in movements.yaml and never implemented -
   compute_mask bounded the current phase's GREEN instead.
6. Five of seven KPIs were computed over COMPLETED trips only, so stranding traffic
   improved a controller's average wait.

The lesson: this project's failures are SPECIFIED-BUT-UNENFORCED invariants, and metrics or
controls that measure something subtly different from what they claim. Hunt for that shape.

UNCOMMITTED CHANGES IN THE WORKING TREE (made 2026-08-30, NOT yet reviewed - audit these
hardest, and update this block when they are superseded):
A. specs/movements.yaml safety.max_red_s: 60 -> 120, with aligned last-resort fallbacks in
   src/env/intersection.py (max_green_s and max_red_s no longer default to each other).
   Rationale: 4 phases => worst-case red = 3*max_green + 4*yellow + 2*all_red = 196s, so a
   60s bound was arithmetically unreachable while max_green_s=60. Measured floor ~95s.
   CHECK: is 120 provably achievable in the WORST case, or only on average? Does the bound
   now actually BIND? Does any test still hardcode 60 and pass for the wrong reason - i.e.
   is a test now passing only because the bound got LOOSER?
B. src/ml/dqn.py learn(): F.mse_loss -> F.smooth_l1_loss(q, target, beta=HUBER_KAPPA).
   Rationale: grad clipping reportedly fired on 100% of updates because the reward is the
   unnormalized -sum|pressure| (~1e4 when jammed), and an always-firing global-norm clip
   divides each update by its own error magnitude, down-weighting the largest-TD-error
   transitions - which are the gridlock events.
   CHECK: verify that 100%/1e4 claim against runs/*/steps.csv (grad_norm is logged pre-clip,
   grad_clip=10.0) rather than trusting it. Then judge sufficiency: Huber's gradient is
   bounded at beta, so if TD errors really are 1e3-1e4 the loss degenerates toward
   sign(error) and can no longer distinguish 'bad' from 'catastrophic' - the exact
   discrimination the gridlock question needs. Is beta=1.0 right at this reward scale?

THE PIPELINE IS ABOUT TO BE RE-RUN (~4h) AND EVERY PRIOR RESULT IS VOID. Weight your
severity by: does this change what gets RECORDED, and can it be retrofitted afterwards?

Report CONFIRMED (traced in code, exact failure statable) vs SUSPECTED. Cite file:line.
No style nits. Severity: critical / high / medium / low.
`

const LAYERS = [
  {
    name: 'simulation',
    scope: 'src/env/**, scripts/build_network.py, scripts/build_routes.py, config/network/**, src/baselines/**',
    hunt: 'right-of-way and physics correctness; gridlock and deadlock; collision handling; ' +
      'determinism; safety-mask correctness and whether the env ENFORCES what it publishes; ' +
      'anywhere a warning or error is suppressed.',
  },
  {
    name: 'rl',
    scope: 'src/ml/**, scripts/train_*.py, scripts/env_factory.py',
    hunt: 'Bellman target and terminated-vs-truncated handling; action-masking integrity; ' +
      'whether the frozen forecaster and the random control are genuinely frozen, ' +
      'scale-matched and information-free; observation bounds; determinism; checkpoint and ' +
      'resume integrity; what happens to a deployed policy on NaN or a bad checkpoint.',
  },
  {
    name: 'systems',
    scope: 'src/api/**, src/db/**, src/trace/**, src/schema/**, frontend/src/**, unity/SmartTrafficViz/Assets/Scripts/**',
    hunt: 'concurrency around the single non-thread-safe TraCI connection; resource leaks on ' +
      'ERROR paths; WebSocket backpressure and reconnection; SQLite/JSONL durability; ' +
      'whether a failed run leaves diagnosable evidence; input validation and path traversal.',
  },
  {
    name: 'statistics',
    scope: 'scripts/eval_runner.py, scripts/analyze_eval.py, scripts/build_analysis.py, src/metrics/**, src/provenance/**, src/repro/**, analysis/final_eval.ipynb',
    hunt: 'every deviation between the code and the FROZEN preregistration.md; how censored ' +
      'episodes are detected and whether the censoring is informative (does which pairs ' +
      'survive depend on the outcome?); whether paired observations are genuinely ' +
      'independent; survivorship bias in any KPI; whether the provenance chain is populated ' +
      'or merely declared.',
  },
]

const requested = Array.isArray(args) ? args.map(String) : []
const layers = requested.length
  ? LAYERS.filter(l => requested.some(r => l.name.includes(r) || r.includes(l.name)))
  : LAYERS

// Phase 1 - one reviewer per layer.
const reviews = await pipeline(layers, layer =>
  agent(
    `${CONTEXT}\n\nYOUR LAYER: ${layer.name}\nSCOPE: ${layer.scope}\n\nHUNT FOR: ${layer.hunt}\n\n` +
      `Return a ranked list of findings, most severe first. For each: severity, file:line, ` +
      `a one-sentence claim, the concrete failure scenario (inputs/state -> wrong behavior), ` +
      `and a fix direction. End with what you checked and found clean, so coverage is visible.`,
    { label: `review:${layer.name}` },
  ),
)

// Phase 2 - adversarial verification. A different agent tries to REFUTE each report, so a
// plausible-sounding finding that does not survive contact with the code is filtered out
// before it reaches the ranked report.
const verified = await pipeline(
  reviews.filter(Boolean).map((review, i) => ({ review, layer: layers[i]?.name ?? 'unknown' })),
  ({ review, layer }) =>
    agent(
      `${CONTEXT}\n\nA reviewer audited the ${layer} layer and reported the findings below. ` +
        `Your job is to REFUTE them. For each finding, read the actual code and decide: ` +
        `CONFIRMED (you reproduced the reasoning and the failure is real), ` +
        `REFUTED (the code does not do what the finding claims - say exactly why), or ` +
        `OVERSTATED (real but lower severity, or already guarded elsewhere - say where).\n\n` +
        `Be skeptical: a finding that sounds right but is not traceable to a specific line is ` +
        `REFUTED. Preserve the file:line citations of everything that survives.\n\n` +
        `FINDINGS TO CHECK:\n${review}`,
      { label: `verify:${layer}` },
    ),
)

// Phase 3 - one report. Deduplicates across layers (the same root cause often surfaces in
// two of them) and ranks by what actually changes a result or breaks in production.
const report = await agent(
  `You are the CTO consolidating a production-readiness audit of ${REPO}.\n\n` +
    `Below are per-layer findings that have each survived an adversarial verification pass. ` +
    `Produce ONE report:\n` +
    `1. A verdict paragraph: is this shippable, and what is the single biggest risk?\n` +
    `2. Findings ranked by severity, deduplicated across layers (the same root cause often ` +
    `appears in two). Keep file:line citations and the CONFIRMED/OVERSTATED verdict.\n` +
    `3. Separate the ones that must be fixed BEFORE any results are regenerated (because they ` +
    `change what gets recorded and cannot be retrofitted) from the ones that can follow.\n` +
    `4. Anything that needs a HUMAN decision rather than a code change - especially any ` +
    `deviation from the frozen preregistration, which requires a dated amendment.\n\n` +
    `VERIFIED FINDINGS BY LAYER:\n\n${verified.filter(Boolean).join('\n\n---\n\n')}`,
  { label: 'consolidate' },
)

return report
