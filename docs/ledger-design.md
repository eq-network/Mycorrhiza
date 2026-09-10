# Ledger coupling — conserved flows as the cross-domain grammar

*Deposited 2026-07-30 from a forest-walk + design session on universal cross-subsystem
influence diagrams. Direction set by Jonas: the κ-gated modulation couplings in
`coupled_society` are a v0 cheat — the proper mechanism is shared conserved ledgers,
built in this library; the playground is a view of the underlying dynamics, never the
home of any. Siblings: `docs/model-register-design.md` (the register pattern this
extends one level up — its §2 verdict structure is reused here), `docs/abm-suite-design.md`
(mechanisms-attach-via-scheduler), `docs/alpha-context.md` (A5 framing),
`docs/observability-design.md` (the run-record path that makes "playground = pure view"
real). Epistemic status: **design, Stage 1–2 (forest-walk output)** — parked behind the
observability layer when deposited, then unparked by Jonas the same day: **v1 shipped
as `environments/ledger_society`** (2026-07-30, 8-rung ladder passing; three channel
dials replacing κ; the knockout instrument and the coupled_society invariance run
remain open).*

## 1. The coupled_society verdict

`coupled_society` is live v0 and its ladder passes — the verdict is about what its
couplings *are*, not whether they run. Findings, against
`environments/coupled_society/dynamics.py:71-160`:

- **Three of five channels mint influence for free.** `economic_power_buys_persuasion`
  and `persuasion_shifts_politics` boost effort/attractiveness as a *function of* an
  income share without any income being spent; `politics_rewrites_market_rules` erodes
  enforcement from an influence reading, costlessly. No opportunity cost means GD §5's
  "General Incentives" question — where do actors route resources when influence in
  another domain pays? — is inexpressible in-model. Only `regulatory_capture` and
  `converts_capitalize` move actual money, and they are the template, not the problem.
- **Beneficiaries are hardcoded.** Every channel tests `node_types == 1` (dynamics.py:81,
  95, 130, 150). The channels are AI-only by construction, so GD §5's central structural
  claim — cross-system channels are *agnostic to human values* — is assumed rather than
  grown. A human plutocrat cannot buy reach; the falsifying experiment cannot be run.
- **κ has no units.** One scalar gates five heterogeneous channels; the gains are
  hand-set multipliers with nothing to anchor them. The env's own assumptions card
  already confesses this ("linear-in-κ with hand-set gains — chosen for legibility, not
  measured; the gains' magnitudes ARE the result's scale").
- **Kept, on merit:** the same-key twin instrument discipline and the sealing identity
  as a ladder rung; the two conserving channels (one of them — `converts_capitalize` —
  per-body, not aggregate); the schedule as the composition operator; the mild-domains
  premise (per-domain recoverability, so joint decline is attributable to coupling);
  the (N, 3) multi-domain action boundary.

**Decisions.** (a) `coupled_society` is **demoted to the pedagogical rung** and stays in
the registry — the same move as `compute_economy`, one level up. (b) The flagship
combined scenario rebuilds as **`ledger_society`**, a fork per §7 of the register doc.
(c) The claim discipline extends to the coupling layer: the defense-transfer-gap
finding is headline-worthy only if it survives **both** coupling implementations
(κ-modulation and ledger) — coupling-implementation invariance, the register's §1
triangulation logic applied to couplings themselves. Execution of the demotion (card
status lines, benchmark wiring) waits for the build session, per the equivalence
discipline in CLAUDE.md.

## 2. What a ledger is

A ledger is an ordinary evolving `node_attrs` field carrying a **conserved stock** —
`wealth` first — with a declared conservation contract. Deliberately not a framework:

- **Declaration:** a module-level dict in the environment,
  `LEDGERS = {"wealth": {"sources": ("production",), "sinks": ("consumption",)}}`.
  Sources mint, sinks burn, both named; every *other* transform that writes a ledger
  field must move value (sum-preserving to float tolerance).
- **Enforcement:** a behavioral-test helper (one function, `environments/ledger.py`):
  run the pipeline one tick, assert each ledger's sum changed only by its named
  sources/sinks. A validation-ladder rung, not a runtime check — JIT stays clean.
- **Metadata:** `system_graph()` tags ledger fields, so the derived DAG can type its
  edges — a write to a ledger field is a *flow* edge, a write to a rate/param field is
  a *modulation/capture* edge. Edge types derived from metadata, never drawn.
- **The legibility lint (near-decomposability by construction):** ledgers and declared
  ports (§5) are the **only** state a domain's transforms may read from another domain.
  A cross-domain read of anything else fails the graph check. This is what keeps the
  coupling surface bounded as subsystems are added: it grows in ledgers × channels, not
  states².

Precedent already in the codebase: the political substrate's row-stochastic `listening`
matrix *is* a conserved attention ledger — each human's outgoing weights sum to 1 and
`rewire` reallocates rather than mints. Money conservation is the missing twin, not a
new idea.

## 3. Allocation is the coupling

The grammar's core move: **cross-domain influence is never a modulation — it is
spending.** Each actor allocates this tick's income across channels:

    consume | reinvest | broadcast (persuasion) | lobby (rule pressure)

- **v0 is fixed allocation fractions** ("the stupid thing first") carried on the action
  channel — the boundary already takes (N, 3) actions in `coupled_society`;
  `ledger_society` takes an (N, K) allocation vector. Adaptive portfolio rules are the
  learning-agent backlog item, explicitly not v1.
- **Every conversion rate has units.** `reach_per_spend` (broadcast effort per unit
  wealth), `pressure_per_spend` (enforcement movement per unit lobbying flow),
  replacing κ·gain pairs. Still toy dials — but dials with units have empirical
  handles: advertising intensity as a share of revenue, lobbying expenditure data,
  and the money→policy elasticity literature (Gilens–Page 2014 — **contested**: the
  Enns/Bashir reanalyses find much smaller elite–median differentials; usable as a
  dial-range debate, not a settled anchor). The magnitude criticism on the current
  card gets an answer instead of a shrug.
- **Value-agnostic by signature.** No channel tests `node_types`. A human windfall buys
  reach; an AI actor that doesn't spend captures nothing. GD §5's "channels are
  agnostic to human values" becomes a modeled fact, with two falsification experiments
  shipped as tests: (i) grant one human a capital windfall → their attention share must
  rise; (ii) zero one AI actor's influence allocations → its capture must vanish.
- **The flywheel assembles itself.** When attention and rules respond to spending, and
  human income share falls, human political weight follows *without any authored
  economy→politics arrow* — coupling emerges from conservation + allocation, which is
  the answer to "how do we get influence diagrams without handcrafting every edge."
  Disempowerment then requires AI actors to actually out-accumulate and out-spend —
  a dynamic to demonstrate, not an assumption to encode.

## 4. The institutional residue

Two couplings are not conserved flows and stay authored — this is the honest size of
the handcrafted layer:

- **Capture:** `enforcement` remains a slow global variable, but moved by *net lobbying
  flow* (funded pressure from whoever spends, in whatever direction) instead of read
  off the human influence share for free. Value-agnostic: citizens' lobbying defends
  enforcement exactly as AI capital's erodes it.
- **Rewiring:** paid reach enters `rewire`'s attractiveness term as bought attention —
  broadcast spend competing for the conserved attention ledger, displacing organic
  weight.

Both are now driven by ledger flows (spend in → effect out), so their parameters are
conversion rates with units. A state actor remains absent (open question §9 — the
taxation–representation mechanism wants one eventually).

## 5. Ports and the derived influence diagram

Not every legitimate cross-domain read is a flow (`enforcement` is an institutional
fact, not a stock). Non-ledger cross-domain reads go through **ports**: declared
order-parameter fields written by an explicit reduction transform (`enforcement`,
`converted_share`, …). Ports are the Flack/Simon boundary — subsystems couple through
low-dimensional coarse-grainings or conserved stocks, nothing else.

With domain tags on transforms, the **subsystem influence diagram is the quotient of
the derived system DAG by domain**: collapse each domain's internals to one node, keep
crossing edges — which are all typed already (flow via ledger, capture/modulation via
port, rewiring via adjacency write). The GD mutual-reinforcement figure becomes a
derived artifact any coupled model gets for free, rendered by the existing
PipelineScene with zero per-game code. `system_graph.py` grows a `domain=` tag and a
`quotient()` — an extension, not a rewrite.

## 6. Instrument migration

The κ=0 sealed twin does not survive — "money doesn't exist" is not a counterfactual.
Its replacements are sharper:

- **The sealing identity relocates — as a design conjecture, not a property.** With
  all cross-domain allocation fractions at zero, no spend occurs, so no cross-domain
  read fires; the target rung is culture/politics matching the solo substrates under
  the same key. Caution: the current κ=0 identity is bit-exact *because the pipeline
  shape is identical*; the ledger version compares **different pipelines**, so
  bit-identity requires engineered RNG-threading discipline that may not be worth its
  complexity — the rung must either define that discipline or honestly weaken to
  statistical equivalence. The null's meaning improves either way: "no influence
  spending" is GD §5's actual counterfactual.
- **Per-channel knockout.** Zero *one* allocation channel per twin (common random
  numbers, paired-shift discipline as in `labor_dependence`): the composite change is
  that edge's measured contribution. The knockout matrix over channels is the
  **empirical A** for the spectral thread — T=(I−A)⁻¹ and the ρ(A) margin get numbers
  derived from the model instead of a posited linearization. This is hypothetical
  extraction (io_economy's validation trick) generalized to the coupling layer.
- **Influence-from-birth vs influence-now**, mechanized: a clamped-read twin (politics
  reads t=0 wealth forever) is the "system responds to who you were" counterfactual —
  the existing instrument-design distinction gets a direct implementation.

## 7. Complexity guards

Realism must not multiply illegibility; the guards are structural:

- Conservation makes bugs self-announcing (a ledger test fails loudly; a wrong
  modulation just looks like a different world).
- Fixed fractions before adaptive portfolios; heterogeneity only where the phenomenon
  lives (wealth and culture distributions, scalar elsewhere).
- **Phase diagrams over (allocation fractions × conversion rates), never single
  trajectories** — the Gualdi–Bouchaud discipline already adopted by the register.
- One screen of dynamics per module stays the bar; the ledger helper is a test
  utility plus a dict convention, not an abstraction layer.

## 8. Sequencing and the playground correction

- **The coupling never lived in the playground.** The live page is a parity port of
  `environments/coupled_society/` in `apps/playground/src/engine/kernel.js`
  (static-host constraint, by design). The debt is the hand-port itself — which is
  verbatim the observability layer's "why-now" (fixtures drift every time a model
  changes). `ledger_society` therefore ships **backend-first**; its web view arrives
  through the run-record path (O2), not another hand port. The playground-as-pure-view
  desire *is* the observability plan — one build, not two.
- **Superseded 2026-08-01, by Jonas's direction:** the coupled scenario's port was
  swapped to `ledger_society` ahead of O2, replacing the κ port rather than adding to
  it. Recorded as an exception in CLAUDE.md's boundary section together with the three
  things it was required to carry (measured parity, a generated system-graph fixture,
  engine-defined channel series). The non-goal below is therefore spent; the
  observability plan is still what retires the port.
- **Active intent holds:** observability (O0/O1) first; new environments stay parked.
  This doc is the R1-style deposit that makes the build cheap when its slot arrives.
- Suggested kanban items (transfer by hand): `ledger_society` env (fork of
  coupled_society; conservation rungs + agnosticism tests in the definition of done);
  `environments/ledger.py` conservation helper; `system_graph` domain tags +
  `quotient()`; knockout-matrix → ρ(A) analysis module (`lab/analysis`);
  coupling-implementation invariance run (transfer gap under both implementations).

## 9. Open questions and non-goals

- **Economy leg:** fork from `compute_economy` (smallest change) or `io_economy`
  (demand attribution gives consumption a real sink and the spectral operator comes
  native)? The register's answer may be "eventually both" — that is the point of it.
- **Attention as an explicit second ledger** with broadcast spend buying share directly,
  vs. the current attractiveness-mediated route — the conserved structure is already
  there in row-stochastic `listening`.
- **The reservoir's insularity, and the floor it was hiding.** *Measured 2026-08-01,
  8 seeds, no CI — a floor, so the claim is its existence, not its size.* Freezing AI
  rows in both adjacency ledgers (`frozen_rows=node_types == 1`) pins them to the t=0
  Erdős–Rényi draw, and that draw points them mostly at the **human** block: an AI row
  sends ≈0.69 of its attention and ≈0.58 of its ballots to humans and keeps doing so
  for the whole run. Consequence: the human attention share could not fall below
  ≈0.45 under *any* dial setting — maximum channel dials, zero diagonal floors, zero
  repair, all landed on 0.4505. That is a property of the initial draw, not of the
  coupling, and it had been reading as a result. Fixed by `ai_insularity` (§ config),
  which redirects a share of each frozen row into the AI block; 0 is bit-identical, 1
  makes the AI block a closed recurrent class so the human attention share goes to 0.
  With insularity, maximum dials, no repair and no diagonal floors the composite
  reaches 0.003 — full capture is now expressible, which it previously was not.
  **Open:** insularity is a workaround for the frozen row, not a resolution of it. The
  deeper question is whether the reservoir idiom belongs here at all — letting AI rows
  drift under the same attachment kernel would make "who the AI attends to" endogenous
  instead of a parameter. That is a bigger change and wants its own equivalence run.
- **A state actor** and enforcement as *funded capacity* (state revenue → enforcement
  budget): makes GD's taxation–representation mechanism mechanical, but adds an actor
  class; post-v1.
- **Schedule × ledger:** spends between domain firings hit stale prices — lag structure
  emerges from the schedule instead of being assumed. First scan repeats the {1,2,4}³
  cadence sweep on the ledger implementation.
- **The invariance run is the exit criterion:** if the defended-gap-exceeds-undefended
  finding survives the implementation swap, it graduates from artifact to claim; if it
  doesn't, that is the more valuable result and gets reported first.

  **First result, 2026-08-01 — it does not survive, and the run is not yet clean.**
  *Measured, n = 64 paired seeds (common random numbers), T = 400, late window.*
  Transfer gap (composite sealed − composite coupled, at a fixed defense setting) is
  **0.2420 ± 0.0045 undefended and 0.0876 ± 0.0033 defended**; the paired difference
  gap(defended) − gap(undefended) is **−0.1545 ± 0.0032** (95% t-interval), and **0 of
  64 seeds** fall on the other side. Under κ-modulation the same difference was
  reported positive (0.128 > 0.109, 8 seeds, no CI). So the sign flips, and not
  marginally.

  **What this does not yet establish.** The two runs do not hold the defenses fixed:
  κ's were attached mechanisms (`aiTax`, `sortition`, `influence_cap`), the ledger's
  are in-transform dials (`reach_cut`, `churn`, `repair_rate`). Two things changed at
  once, so the reversal is not yet attributable to the coupling implementation rather
  than to the defense instruments. **A clean run needs defenses that exist in both
  models** — the obvious candidate is a tax defense, since both models tax, plus a
  reach cap expressible as a mechanism in the κ pipeline. Until that exists, the
  honest statement is the weaker one: *the finding is implementation-sensitive*, which
  is already enough to keep the κ-era transfer-gap numbers out of any claim.
  Recorded rather than retuned, per WP1 §5.
- Non-goals: adaptive allocation policies (learning-agent backlog); real-world
  calibration (dials get units and order-of-magnitude anchors, not fits); any change
  to the frozen `GameSpec` boundary. *(The fourth non-goal — touching the playground
  before O2 — was spent on 2026-08-01; see §8.)*
