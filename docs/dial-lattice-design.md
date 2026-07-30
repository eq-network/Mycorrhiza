# Dialed experiments on the web — the sweep-bundle / lattice plan

*Deposited 2026-07-30, direction set by Jonas: visitors should be able to set
parameters and sweep the GD-suite models "in real time or something like it,"
playground-style, without violating the engine⇄view boundary. Siblings: CLAUDE.md
⟦BOUNDARY⟧ (no hand-ports, crossings are versioned artifacts),
`docs/web-trajectory-contract.md` (the payload format, v1.1),
`docs/observability-design.md` (O0/O1 manifests and metric identity — this plan is a
consumer of those, not a rival scheme), `docs/ledger-design.md` §6 (the knockout
instrument that rides the same sweep). Status: **plan approved, in implementation**
(2026-07-30, plan file `okay-implement-it-then-inherited-nest.md`). Review-pass
deltas: derived (T,) series computed engine-side in the exporter (contract stays
v1.1, tick stride 1); `adj` omitted from v1 payloads; one rectangular lattice per
bundle (presets are later bundles); schema enforced twice — Python `jsonschema` on
write, `ajv` in eq-network CI.*

## 1. The move

The slider lattice, the phase map, and the knockout measurement are one object.
House discipline already forbids reporting single trajectories — the deliverable is
phase diagrams over the coupling dials — and `ledger_society` exposes exactly three
channel dials (`reach_per_spend`, `attention_to_ballots`, `regime_rate`). So:

    engine sweeps the dial lattice offline (vmap; minutes at N=26, T=300)
      -> one versioned static bundle
      -> the web page is SELECTION over real runs, never computation

"Real time" decomposes into two honest tiers:

- **0 ms — the scalar layer.** All lattice cells' late-window scalars with
  bootstrap CIs (~343 × 10 numbers ≈ 30–50 KB) load upfront; dial-dragging snaps to
  the nearest cell and updates every gauge and heatmap instantly.
- **~100 ms — the playback layer.** "Run it" lazily fetches that cell's
  representative trajectory (one seed, contract v1.1 JSON, ~40–80 KB gzipped,
  cached) and replays through the existing `Trajectories.fromJSON` path — which
  already carries the `system` DAG, so the derived pipeline view is free.

Off-lattice, truly arbitrary parameters have exactly two honest routes, both
deferred: the `RemoteEngine` endpoint the trajectory contract anticipates (a
server — an ADR decision, not a default), or literal "on their own computers":
`pip install` plus a marimo/Jupyter example with live sliders driving the real
engine (cheap; serves researchers and forkers; the web serves visitors).

## 2. Phases

- **L0 — freeze the dial set.** The three channel gains; `first_arrival` as a
  fourth axis or a preset toggle. Start 2 dense axes × 1 coarse, not 7³ everywhere.
  Every dial published must be typed on the assumptions card (the lattice IS the
  "arbitrary-but-swept" receipt).
- **L1 — the sweep-bundle format.** A versioned static directory:
  `manifest.json` (env, config hash, seed batch, code version, metric ids +
  definition hashes), `scalars.json` (the lattice), `runs/<cell>.json`
  (contract-v1.1 trajectories, one representative seed; optionally only a
  sub-grid carries playback). **Constraint: this must be an O0/O1 run-record
  artifact, not a parallel format** — metric identity and manifest schema come
  from the observability layer, or the bundle recreates the fixture-drift problem
  with a new extension. If O0/O1 are not yet frozen when L1 starts, L1 is the
  forcing function to freeze them.
- **L2 — the generator.** `experiments/ledger_lattice/` in the `_template` shape:
  `config.py` freezes the grid and committed expectations, `run.py` sweeps with
  bootstrap CIs, `export.py` writes the bundle. The knockout/ρ(A) analysis
  (ledger-design §6) consumes the same sweep — per-dial-sealed cells are lattice
  edges, so the empirical coupling matrix comes out of the identical run set.
- **L3 — the web consumer.** A NEW unlisted page (never the frozen
  `playground.html`): dials → phase heatmap with current cell marked → vitals
  playback → derived system view. Generic over bundles — the next environment
  gets dialed sweeps by publishing a bundle, zero per-game code. Reuses the
  fromJSON/PipelineScene logic as proper components (a port of the *shell*, which
  the boundary permits — it is view code, not dynamics).
- **L4 — optional, later.** The marimo local-live example; the RemoteEngine
  endpoint (needs the eq-network no-backend ADR revisited; CORS service separate
  from the static site).

## 3. Sizing (napkin, honest)

343 cells × 8 seeds: ~2,700 runs, vmap-batched — minutes of laptop compute.
Scalars: tens of KB. Playback: 40–80 KB/cell gzipped after field pruning
(playback needs ~half the trace fields), T-subsampling ×2, and float rounding;
125-cell playback sub-grid ≈ 5–10 MB total, fetched lazily so nobody downloads
more than the cells they visit.

## 4. The full-stack build (L3 made concrete, 2026-07-30 addendum)

**Stack: what already exists is enough.** Astro 4 + React on GitHub Pages; the new
page is one Astro route + one React island; bundles are static files in
`public/`. No new framework, no backend, no build tooling beyond what ci.yml runs.

```
eq-network/apps/site/
  public/lab/runs/ledger-society-v1/     # the "database" (static, versioned)
    manifest.json                        # env, config hash, code version, metric ids
    scalars.json                         # the full lattice, CIs included (~50 KB)
    runs/c-<i>-<j>-<k>.json              # contract-v1.1 trajectory per cell (lazy)
  src/pages/lab/dials.astro              # the route (unlisted, like /lab)
  src/components/lab/sim/                # the path the contract doc anticipated
    contract.ts                          # TS types for manifest/scalars/trajectory
    engineIO.ts                          # fetch + fromJSON + LRU cache + prefetch
    latticeStore.ts                      # current cell, playback tick (useReducer)
    Dials.tsx  PhaseMap.tsx  Vitals.tsx  # sliders (snap), heatmap+CI, line charts
    PipelineScene.tsx  sketch.ts         # ports of the VIEW code (legal: not dynamics)
    DialLab.tsx                          # the island: <DialLab client:load bundle=.../>

cilib/experiments/ledger_lattice/
  config.py  run.py  export.py           # grid frozen, sweep w/ CIs, bundle writer
  schema/bundle.schema.json              # THE shared contract artifact
```

**The contract is a schema, enforced twice.** `bundle.schema.json` lives in cilib
next to the exporter (which validates on write); eq-network CI gets a node step
that validates every bundle in `public/lab/runs/` against the same schema and
checks manifest hashes. Cross-repo drift then fails a build instead of a demo.

**Runtime data flow.** Page load: manifest → scalars (
gauges + phase map render immediately). Dial drag: snap to nearest cell, update
from memory (0 ms). Cell select/play: fetch `runs/c-….json` (~40–80 KB, Pages
gzips JSON), LRU-cache, `requestIdleCallback` prefetch of neighbor cells so
slider-walking feels live. Playback: rAF loop over the trajectory driving the
sketch scenes — same rendering idiom as the playground, now fed by real runs.

**Hosting ladder.** v1: bundles committed in `public/` (MBs are fine on Pages; no
LFS — Pages doesn't serve it). v2 (many bundles): move to a data home —
`runs.eq-network.org` on Cloudflare R2 or a separate gh-pages data repo, CORS
pinned to the site origin; the page takes `bundleUrl` as a prop so this is a
config change, not a rewrite. v3 (off-lattice live dials): `RemoteEngine` —
FastAPI + cilib in a container (Fly.io/Cloud Run), warm compiled step-fns per
env, `POST /run {env, params, seed}`, responses cached by (config-hash, seed);
requires revisiting the site's no-backend ADR, costs ~$5–15/mo warm or 2–5 s
cold-started free.

**What is ported vs. banned.** The sketch renderer, scene shell, and
`Trajectories.fromJSON` are *view* code — porting them into `sim/` is legal and
finally gives them a typed, module home. Dynamics are never ported;
`playground.html` stays frozen and keeps its self-test CI gate. Optional later:
a build script that inlines one small bundle into a single self-contained HTML
for the private-artifact publishing path the prototype used.

**First PR slice (a weekend of focused work):** contract.ts + engineIO + Dials +
PhaseMap against a hand-rolled 3×3 toy bundle from the exporter; playback and
PipelineScene in the second slice; CI schema step with the first real bundle.

## 5. What this is not

- Not a JS port: the page computes nothing; the ⟦BOUNDARY⟧ holds by construction.
- Not the observability layer itself: O0/O1 stay the prerequisite and keep their
  queue position — this plan is their first paying customer on the web side.
- Not a claim vehicle: lattice scalars ship with CIs and the card's sign-only
  discipline; the phase heatmap displays uncertainty, not just color.
