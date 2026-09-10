# Observability & live-runner design — the state protocol

Status: **design record, not yet built** (2026-07-28). Frame: the
visualisation-and-observability design session and its strategy note. Maps onto
the observability epic items **O0–O6**; this file refines them, it does not
introduce a parallel scheme.

## Purpose (first principles)

A run of this library is **not an event**. The scan tier is pure and seeded, the
tick lives in `global_attrs["step"]` rather than in the loop, and `close()`'s
`round_fn(state, t, key)` ignores `t` entirely — the environment's own step
counter does the work. So a `GraphState` is a **sufficient statistic**: the
future depends on the past only through it (the Markov property).

Everything below is a consequence of taking that seriously. If a run is
reproducible from a description, then the durable artifact is the
**description**, not the trajectory — and the whole storage design inverts
relative to an ML experiment tracker.

## Core abstraction

```
Address  ::= (env_name, config, [MechanismSpec], key, EditLog, code_version)
State    ::= GraphState                                  -- the sufficient statistic
Record   ::= (Address, Manifest, {MetricId: Scalar})     -- what we STORE

replay   :: Address -> State                             -- pure, deterministic
snapshot :: State -> Bytes                               -- tree_flatten
restore  :: Bytes -> State                               -- tree_unflatten
project  :: Array -> Array                               -- constraint repair
```

**Laws** (each is a cheap test, and each is load-bearing):

```
restore ∘ snapshot          ≡ id                  -- serialisation is lossless
replay(addr ⊕ e)            ≡ apply(e, replay(addr))   -- edits stay addressable
project ∘ project           ≡ project             -- repair is idempotent
project(valid)              ≡ valid               -- and a no-op where valid
replay(addr)[0:t]           ≡ run(addr)[0:t]      -- pausing changes nothing
```

The last one is the proximate objective in one line.

## The consequence that drives the design

An **edit breaks the address chain** — an edited state is no longer derivable
from `(config, key)`. There are two ways out, and they are not equivalent:

- **Store the edited state.** Simple, and it makes every edited run an opaque
  binary blob whose provenance stops at the edit.
- **Store the edit as data.** An `EditLog` of typed operations appended to the
  address. The run stays replayable from `(seed address + edit log)`, and
  nothing needs to store state at all.

**Take the second.** It preserves addressability through the editor, which is
the property the entire storage design rests on. Snapshots then exist only as a
*cache* (skip re-simulation of a long prefix), never as the source of truth —
and a cache can be deleted without losing anything.

## Functional decomposition

```
                        ┌──────────────────────────────┐
   PURE CORE            │  existing, unchanged          │
   (compiled, seeded)   │  compile_pipeline · run_scan  │
                        │  GameSpec/close · reducers    │
                        └───────────────┬──────────────┘
                                        │ GraphState
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
      ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
      │ core/serialize   │   │ core/edit          │   │ metrics/identity │
      │ snapshot/restore │   │ EditOp · project   │   │ MetricId + hash  │
      │  (pure)          │   │  (pure)            │   │  (pure)          │
      └────────┬─────────┘   └─────────┬──────────┘   └────────┬─────────┘
               └───────────────┬───────┴───────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ runs/  (record I/O) │   ← EFFECT BOUNDARY
                    │ Address·Manifest    │      filesystem only
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
      ┌─────────────────────┐      ┌────────────────────────┐
      │ apps/runner (local) │      │ examples/05 exporter   │
      │ chunked loop, socket│      │ v1.1 JSON recording    │
      │  (effectful)        │      │  (effectful)           │
      └─────────────────────┘      └───────────┬────────────┘
                                               ▼
                                        static lab page
```

## Module breakdown

### `cilib/core/serialize.py` — NEW, small

- **Signature**: `snapshot :: GraphState -> bytes`, `restore :: bytes -> GraphState`
- **Purity**: pure
- **The point**: this is *nearly already written*. `tree_flatten` returns
  `(children, aux_data)` with keys **sorted** — a canonical, deterministic
  decomposition into a flat list of arrays plus a small static tuple. Encode
  children as `.npz`, `aux_data` as JSON, add a `protocol_version` int. Restore
  is `tree_unflatten`. Expect ~40 lines.
- **Do not** invent a schema. The pytree registration *is* the schema; a second
  description of the state layout is exactly the drift this repo keeps deleting.
- **Watch**: `aux_data` carries static Python scalars from `global_attrs`; those
  must round-trip by value. Sparse BCOO children need their own branch
  (`indices`/`data`/`shape`), since they are not plain arrays.

### `cilib/core/edit.py` — NEW, small

- **Signature**: `apply :: (EditOp, GraphState) -> GraphState`
- **Purity**: pure. Runs host-side, outside the traced domain.
- **Scope discipline**: three typed ops to start, one frozen dataclass each —
  `SetEdge(matrix, i, j, value)`, `SetNodeAttr(name, index, value)`,
  `SetGlobal(name, value)`. Readable in one screen. **Resist an edit DSL**;
  add the fourth op when a real need appears, not before.
- **Constraint repair** is a companion, not a special case:
  `project :: str -> (Array -> Array)`, resolved from a plain dict colocated
  with each environment (the `ASSUMPTIONS.md` precedent — env knowledge lives
  with the env):

  ```python
  CONSTRAINTS = {"listening": row_stochastic, "friendship": symmetric_binary}
  ```

  `apply` is then `project ∘ raw_edit`. This is where the `influence_exchange`
  hazard is handled once instead of in every client: zeroing an entry of a
  row-stochastic matrix silently redistributes attention unless renormalised.
  Constraints are **retractions** — idempotent, and the identity on valid
  states — which makes them two-line property tests.

### `cilib/metrics/identity.py` — NEW, tiny (epic **O0**)

- **Signature**: `MetricId ::= (name, kind, version, source_hash)`
- `kind` is a closed vocabulary of ~5: `share` · `index` · `level` · `rate` ·
  `flag`. This is what makes **O3** possible — one renderer per kind, not per
  metric, so a new model's readouts chart correctly with no frontend work.
- **Versioning**: carry *both* an explicit `version: int` (authoritative) and a
  `source_hash` (advisory). A hash alone false-alarms on comment edits; a
  version alone gets forgotten. The hash's job is to *warn that a bump was
  missed*, not to identify.

### `cilib/runs/` — NEW package (epic **O1**)

- **Signature**: `write :: Record -> Path`, `read :: Path -> Record`,
  `index :: Path -> [Manifest]`
- **Purity**: effectful — **the only filesystem boundary in the design**.
- **Storage shape**: a directory per run; `manifest.json` (address + git SHA +
  `cilib` version + metric ids) and `scalars.json` are small and belong in git;
  optional `snapshot.npz` is a cache and is gitignored (`results/` already is).
- **Not a database.** Directories, JSON, npz. If an index file ever feels slow,
  that is a signal to look at run counts, not to add a query engine.

### `apps/runner/` — NEW, **outside the installed package**

- **Signature**: `Session ::= Address -> Stream (State, Trace)` over a WebSocket
  (pause/step/edit/resume are inherently session-ful).
- **Purity**: effectful. Deliberately outside `src/cilib` so the **library never
  gains a web-server dependency** — the lab razor: a socket server is not a
  library API. (`apps/` has precedent: `archive/Visualiser`'s `apps/studio/`.)
- **Hard rule**: the runner **calls `run_scan` in chunks**; it never
  reimplements the round loop. The moment it owns its own stepping logic, the
  JS-port problem has been recreated inside Python.

### Extensions to existing modules — deliberately minimal

| File | Change | Size |
|---|---|---|
| `core/scan.py` | return the final carry key (currently `(final_state, _)`) | 2 lines |
| `core/graph.py` | `create_padded_state` + `get_active_mask` — **referenced in `scan.py`'s docstring but do not exist** | small |
| `examples/05_export_trajectory.py` | export from a `Record` rather than an ad-hoc run | moderate |
| `environments/spec.py` | nothing — `run_scan` already takes an initial state | — |
| `environments/game.py` | **nothing.** `GameSpec` is frozen and stays frozen | — |

## Data flow: the live loop

```
Address ──replay──▶ State₀
                      │
        ┌─────────────▼──────────────┐
        │ run_scan(round_fn,         │   compiled once per (shapes, round_fn,
        │          state, CHUNK, key)│   CHUNK) — chunk size MUST be constant
        └─────────────┬──────────────┘
                      │ (state', trace_chunk, key')
                      ├──────────────▶ push trace to client, append to session
                      ▼
                   paused?  ──no──▶ loop
                      │ yes
                      ▼
        device_get(state')            ← leaves the traced domain; free
                      │
              EditOp from client
                      ▼
        apply = project ∘ raw_edit    ← invariant repair happens HERE, once
                      │
              append EditOp to EditLog
                      ▼
        jnp.array(...) ──▶ resume with same compiled kernel
                           (values changed, shapes did not ⇒ no recompile)
```

**The recompilation cliff, stated plainly** — this is the cost model the whole
interface must respect:

| Edit | Recompile? |
|---|---|
| edge / attribute **values** | no — same shapes |
| number of agents (N) | **yes** — unless padded capacity + active mask |
| pipeline membership | **yes** — new `round_fn` |
| chunk size | **yes** — `n_steps` is the static scan length |

That last row is a real trap: a variable chunk size pays a compile per distinct
length. Fix the chunk.

## How the public export stays a strict subset

The exporter becomes a **function of the session record**, not a parallel path:

```
export :: Record -> v1.1 JSON
export = decimate ∘ trace ∘ replay
```

Because the public payload is *derived* from the same `Address` the live runner
used, "strict subset" is structural rather than a discipline someone has to
remember. Two consequences:

1. **The JS ports can be retired.** They exist only because the page needed to
   *compute* rather than *replay*. Once every scenario the page shows is a
   recording produced by the real engine, the second implementation has no job.
2. **O4 (payload budget) mostly dissolves.** You do not ship a decimated
   trajectory hoping it is enough — you re-derive exactly the view requested,
   at the resolution requested, because you hold the address.

Deployment stays honest: the runner is **localhost only** (research/authoring);
eq-network is a static Astro site and consumes recordings. Two surfaces, one
engine, one protocol.

## Critical path

```
   [S1] core/serialize ──┬──────────────────────────────┐
                         │                              │
   [S2] return key ──────┼──▶ [R] apps/runner ──▶ [O2] frontend connection
        (2 lines)        │         ▲
                         │         │
   [E] core/edit ────────┴─────────┘
        + CONSTRAINTS

   [O0] metric identity ──▶ [O1] runs/ ──▶ [O3] kind→chart
                                    ▲
                                    └── [R] feeds it

   [P] padded capacity ──▶ node add/remove ──▶ graph editor (nodes)
                       └─▶ variable-N sweeps without recompiles
```

- **S1 is the keystone.** Four consumers (runner, editor, checkpoint, export)
  and nothing else can start without it.
- **S2 is two lines** and gates the "bit-identical" half of the objective.
- **P is a double unlock** — node editing *and* variable-N sweeps — but it is
  **not on the critical path**, because zeroing an edge changes no shape. Defer
  it; do not let its appeal reorder the work.
- **O0/O1 run in parallel** with the runner branch; they share only `Record`.
- **The graph editor is last on purpose.** It is downstream of S1, E, and (for
  node ops) P. Built earlier, it gets built twice.

## Risks

1. **The runner becomes a second source of truth.** The exact failure the JS
   ports already represent, relocated. *Mitigation:* the runner may only call
   `run_scan`/`EnvSpec.run`; if it ever needs to know what a round does, the
   design is wrong.
2. **Over-abstraction.** An edit DSL, a constraint plugin registry, a run
   database — each is a plausible generalisation of a three-case problem.
   *Mitigation:* three ops, a plain dict, directories. CLAUDE.md's razor:
   a catalog readable in one screen beats a clever registry framework.
3. **Protocol scope creep to the eager/LLM tier.** Correct eventually, fatal
   now — the eager tier has no Markov guarantee and would make S1 unshippable.
   Explicitly out of scope.
4. **Snapshot-as-truth drift.** If snapshots become the stored artifact rather
   than a cache, addressability is quietly lost and the provenance story with
   it. *Mitigation:* keep `snapshot.npz` gitignored; if it is missing, the run
   must still reproduce.
5. **Sparse and padded states complicate serialisation** more than the flat
   dict-of-arrays picture suggests. BCOO needs its own encode branch; padding
   adds a capacity/active distinction that every client must respect.
6. **Metric-kind vocabulary set too early.** Five kinds is a guess. *Mitigation:*
   it is a dict, not a type hierarchy — widening it later is a one-line change,
   and the phase-diagram case (`value_contagion`'s four corners) may not fit any
   of the five and should be allowed to stay special until a second example
   appears.

## Trade-offs recorded

- **Edit log over stored states** — costs re-simulation time on load, buys
  provenance and deletable caches. Correct because runs are cheap and
  reproducible; would be wrong in a world where they were not.
- **`apps/` outside the package** — costs a slightly awkward import path, buys a
  library with no server dependency.
- **Constraints colocated with environments** — costs a little duplication
  across envs, buys the editor not needing to know any environment.
- **Two surfaces (local deep tool, static public page)** — costs one export
  step, buys the retirement of the JS ports and an honest split between a
  provisional research view and a published one.

## Open questions

- Does the phase-diagram view generalise beyond `value_contagion`'s two clean
  dials, or is `coupled_society`'s natural primary view something else entirely?
- Does the editor round-trip into source (code-canonical, per the design session),
  or stay a read-only navigation layer over committed pipelines?
- Where does the ensemble live in the protocol — is a `Record` one seed or a
  condition? (The walk argued the *experiment* is the primitive; this document
  has quietly assumed a `Record` is a single address. That needs resolving
  before O1 is written, and the answer is probably that a condition is a set of
  addresses sharing a config with different keys.)
