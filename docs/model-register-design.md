# The economy model register — structural robustness by design

*Deposited 2026-07-24 from a design session on the scientific validity of the economic
scenario. Direction set by Jonas: the alpha's economic claim must rest on a **register
of structurally different substrates**, not one parameterization; environments must be
fork-friendly starting points a smart high-school student can understand. This doc is
the register's map. Siblings: `docs/abm-suite-design.md` (the suite frame — its economy
row is superseded by this doc, see its 2026-07-24 status note), `docs/alpha-plan.md`
(phasing), `docs/alpha-context.md` (public framing), `docs/game-boundary-design.md`
(the frozen `GameSpec` boundary all register entries build against).*

**Status (2026-07-24, same day):** R1 and R2 shipped. The relabel landed
(`labor_dependence` v1 = the static output elasticity, d logY/d logL over a 10-tick
post-shift window from realized paired shifts; the Cobb-Douglas rung now asserts the
instrument recovers α — re-run benchmark: 0.20 undefended vs 0.51/0.54 defended,
recovery 0.38/0.42, ~40% below v0's biased readings). **`io_economy` is live v0**
(6-rung ladder passing): the Leontief fixed point is exact, hypothetical extraction
matches the analytic inverse (the counterfactual instrument's first analytic anchor),
and — the build's finding — **`reinvest_rate` selects between §2's two endpoints**:
hoarded AI margins drain demand (absolute collapse: wage bill 16→~0), while full
reinvestment conserves demand so total activity *grows* 21.3→32.7 as the human
demand-attribution share falls 1.0→0.01 (relative disempowerment — the economy runs
for its own loop; unsupervised AI spending ~12.7/tick). The same `ai_revenue_tax`
catalog entry as `compute_economy`, unchanged, holds the share at ≈0.85 —
cross-substrate mechanism reuse working as designed. Assumptions cards are live for
all built environments. **`task_economy` shipped as a skeleton the same day**
(production core + comparative-advantage adoption, 4-rung ladder): the Baumol regime
produces the *opposite* of disempowerment (frontier capped at 0.6: wage 1.0 → 4.50,
labor share 0.85 — §10's self-critique honored empirically), full automation gives
rise-then-collapse (wage peaks 3.05, then 0 the tick the last tasks fall), and
capability without profitability automates nothing (`adoption_gap_final` readout).
Remaining R3 work: demand side, compute-capital loop, benchmark wiring.

## 1. Why a register, not a model

The claim the economic scenario exists to test — *gradual economic disempowerment is a
real dynamic* — must be an **invariant across structurally different models of the
economy**, or it is an artifact of one model's assumptions. This is the same
triangulation logic as the evidence-standards draft ("Evidence Standards for
Computational Mechanism Design", Hallgren 2026, in prep): vary what you are uncertain
about and trust what survives. That paper varies *agent* models; the register varies
the *substrate* on the other side of the `GameSpec` boundary. A defense's evidence is
strongest when its effect holds across both.

The register pattern generalizes: the cultural subarea later gets its own (epidemic vs
replicator vs DeGroot-variant substrates). This doc builds only the economy's.
**Instantiated 2026-07-24 — `docs/cultural-register-design.md`** (members
`value_replicator` / `value_contagion` / `value_spectral`; the DeGroot member was dropped
as duplicative of A4). Note its honest caveat: the cultural members are *nested* — limits
of one theory — rather than structurally independent as the economy's three are, so its
robustness claim is correspondingly weaker and says so on every card.

## 2. The `compute_economy` verdict

**Direction set by Jonas (2026-07-24):** the A2 headline was overclaimed. The findings:

- **The instrument re-measures a mainline indicator.** The "influence-NOW" score is the
  response of late-run log-output to a work-preference shift — which, by the CES
  identity ∂logY/∂logL = wage·L/Y, is (to first order) the **labor share** read off the
  trace. The counterfactual machinery, load-bearing in the commons, is redundant here.
- **The collapse is assumed, not found.** With `rho=0.5` (σ=2) and a fixed reinvestment
  rule, labor-share decay is what CES *means*. The validation ladder certifies the
  accounting (Cobb-Douglas rung — at ρ=0) but never exercises the disempowerment twist
  (ρ=0.5): the one load-bearing parameter has no rung.
- **It isn't influence.** The model's only human channel is labor supply; there is no
  governance channel, so "influence" degenerates to marginal product, and the defenses
  (tax, cap) are experimenter-imposed schedules no human in the model chooses. What the
  scenario measures is **economic dependence on human labor** — genuinely §2 of Gradual
  Disempowerment, but not an empowerment score.

**Decisions:** (a) relabel the instrument `influence` → `labor_dependence`; (b) demote
`compute_economy` to the register's pedagogical rung; (c) rebuild the flagship economic
scenario as `task_economy` (§4). The renames in `experiments/benchmark/scenarios.py`
and the benchmark README are **deferred to an implementation session** — "labor
dependence" needs a precise definition first, and the equivalence discipline in
CLAUDE.md applies.

**Kept, on merit:** the closed-form threshold — AI capital compounds iff its growth
factor `1 − δ + s(1−τ)r` exceeds 1; under CES, σ>1 ⟺ r stays bounded away from 0 as
C→∞ ⟺ the loop can self-sustain; the tax that opens it is `τ* = 1 − δ/(s·r∞)`. Also
kept: the **influence-from-birth vs influence-now** distinction (the system responds to
who you *were*, not who you *are*) — the instrument design outlives this substrate.

## 3. Three answers to "what is an economy?"

The audience bar for every register entry: **one plain-language paragraph + one screen
of dynamics code**, understandable by a smart high-school student. The ensemble — three
different answers to the same question — is the teachable object, not any single model.

**The one-machine economy** (`compute_economy`, live, demoted). The whole economy is
one machine with two input slots: human work and computer power. Each tick it pays each
input by how much a little more of it would help. AI actors reinvest their earnings
into more computer power. The question: as the computer slot fills, does the machine
still need the human slot? One dial — how easily computers substitute for people —
decides everything. That is both this model's clarity and its known weakness.

**The recipe economy** (`io_economy`, designed). The economy is a network of recipes —
cars need steel, steel needs mining, everything needs decision-work — and at the end,
people buy finished goods. AI enters by *editing recipes*: replacing human
decision-work with AI cognition bought from a new AI industry. Follow the money
backward from final purchases and you can compute exactly what fraction of all activity
ultimately exists to serve humans. Watch that fraction over time.

**The task economy** (`task_economy`, designed — the rebuild flagship). Every job is a
bundle of tasks, and machines learn tasks one at a time as compute grows — a frontier
sweeping down the list. Humans keep the not-yet-automated tasks; if those are
essential, human wages can even *rise* as the frontier advances — until the last tasks
fall. Firms adopt automation when it pays, so a tax changes *behavior*, not just
bookkeeping.

## 4. The register

| Entry | Status | Validation anchor | Disempowerment dial | Unique contribution |
|---|---|---|---|---|
| `compute_economy` | live, **demoted to pedagogical rung** | Cobb-Douglas labor share ≡ α every tick (passing, `compute_economy/tests/`) | `rho` (σ assumed) | teaching example; closed-form threshold τ* |
| `io_economy` | designed | Leontief-inverse identities; hypothetical extraction ≡ the counterfactual instrument in analytic form (a validation rung for `environments/counterfactual.py`) | coefficient-matrix rewiring (σ=0 bracket) | demand attribution — "share of activity serving human demand"; AI's network position; spectral margin natively |
| `task_economy` | designed, flagship | Acemoglu–Restrepo task-model derivations; Baumol bottleneck limit (complementary tasks ⇒ wages rise until near-full automation) | automation-frontier speed (elasticity *emergent*) | disempowerment as process; endogenous adoption ⇒ defense circumvention testable |

Register entries are ordinary catalog entries (`environments/REGISTRY`) — the register
is a *reading* of the catalog, not a new mechanism. All three build against the frozen
`GameSpec` boundary; a benchmark condition can then score one defense across all
substrates (§8).

## 5. Lineage — what each model borrows, and what it refuses

**From Epoch's GATE** (`task_economy`): the task-based automation frontier — the
automatable task fraction advances with effective compute, so aggregate substitutability
*emerges* instead of being assumed — and **endogenous investment** (returns-driven,
with adjustment costs), which is what makes taxes change behavior and circumvention
pressure exist in-model. Refused: GATE's forward-looking optimization solution concept.
Register entries use myopic behavioral rules ("the stupid thing first").

**From the Farmer / complexity-economics lineage** (Macrocosm, INET Oxford): production
networks on IO topology — *where* AI enters the network matters, not just how much
(`io_economy`); the Mark-0 / Gualdi–Bouchaud **phase-diagram methodology** — map
parameter-space regimes and their boundaries, never report single trajectories (§8);
occupational-mobility networks (displacement as a race between automation speed and
retraining speed) — noted, deferred past alpha.

**From Leontief input–output analysis** (`io_economy`): the demand-attribution
decomposition — split final demand into human consumption `d_H` and AI-loop investment
`d_AI`; human-attributable activity is `1ᵀ(I−A)⁻¹ d_H`, and its share of gross output
is *the* missing alignment metric ("is the economy still producing for humans?").
Hypothetical extraction (delete a sector/factor, measure the output drop) is the
counterfactual influence instrument in closed form — the finite-Δ instrument run on
`io_economy` must recover the algebraic answer, which is the entry's validation rung.
Leontief technology is the **σ=0 bracket**: maximal complementarity, the world most
favorable to human indispensability — if disempowerment appears here too (AI spreading
by coefficient rewiring), it cannot be a substitution-assumption artifact. Cautions:
run as sequential rebalancing dynamics (the Farmer COVID-model pattern), not the
classical dynamic-Leontief capital model (dual-instability pathology); fixed
coefficients mean no price-substitution response *by design* — that is the bracket's
point, not a bug.

**Precision note:** the Leontief inverse `(I−A)⁻¹` and the diagnostics thread's
lock-in operator `T=(I−A)⁻¹` (`docs/alpha-context.md` §diagnostics) share the same
mathematics — Neumann series, ρ(A)<1 margin — applied to *different matrices*
(technical coefficients vs cross-domain influence coupling). Same identity, not the
same object; `io_economy` is where the shared machinery gets a substrate whose operator
is the model itself rather than a linearization.

## 6. The assumptions card

Every register entry (eventually every environment) carries an
**`ASSUMPTIONS.md` colocated in its subpackage** — the fork-legibility artifact. A fork
copies the environment directory, so only a colocated card travels with it by
construction; and per CONTRIBUTING.md, code is the source of truth when docs drift — a
central card registry would drift the moment the second entry lands. Seven fields:

1. **What this says an economy is** — one plain-language sentence.
2. **Assumptions** — plain bullets: functional form, behavior rules, what is fixed vs
   endogenous, what is absent.
3. **Classical result reproduced** — the named validation anchor + path to its test.
4. **The dial** — the config field whose value produces the failure dynamic.
5. **Instrument** — pointer to the scenario's declared (channel, outcome,
   counterfactual design) in `experiments/benchmark/scenarios.py`; never a duplicate.
6. **Lineage** — forked from X, changed Y; or "originates here".
7. **Status** — live / demoted / designed, dated, one line why.

First worked card: `src/cilib/environments/compute_economy/ASSUMPTIONS.md`.

## 7. Forking is the community mechanism

Disagreement with a model is the *intended* interaction: fork the environment
directory, change the assumption, register under a new name (one `REGISTRY` line, per
EXTENDING.md), re-run the benchmark. A fork is **evidence** — a new point in substrate
space — only under two invariants:

- **The instrument contract stays fixed.** The (channel, outcome, counterfactual
  design) triple declared per scenario is what makes fork scorecards commensurable. A
  fork that changes the *instrument* is a **measurement fork** — a disagreement about
  what disempowerment means, not how economies work — and its card must say so
  (field 6). Dynamics forks and measurement forks answer different questions.
- **The scorecard schema stays fixed**, so "run on the new model" auto-produces
  comparable rows.

The register's three entries are the seeded starting points; the assumptions card is
the fork's diff declaration ("think recipes can change? fork the recipe economy and
add substitution — card: forked from `io_economy`, changed fixed coefficients").

## 8. Resilience is the acceptance criterion

*(Research thread, Stage 1–2 — same epistemic status and same discipline as the
spectral-margin thread in `docs/abm-suite-design.md`: the benchmark ships its main
indicators until this survives the literature. Recorded here as the destination.)*

The evidence-standards draft supplies the evaluation shape the register plugs into:

- **Basin-stability maps, not point scores.** Menck's protocol (draw N perturbed
  initial conditions, simulate, count the fraction returning to the desirable regime)
  is one `vmap` axis over `run_scan_batch` — cheap in JAX, and the number it yields is
  a *volume*, robust to the arbitrary choices (Δ, t₀, windows) that made "0.34"
  fragile. The standard deliverable per scenario becomes **basin fraction as a function
  of the mechanism dials**; phase boundaries are where it crosses ½. This goes into
  `task_economy`'s definition of done, not in afterward.
- **Naming reconciliation:** `experiments/basin_stability/` (April 2026) already uses
  the term for a *different* protocol — survival probability swept over the
  adversarial-agent **fraction** (a parameter axis, Wilson CIs). Both are basin
  measurements on different perturbation axes: parameters there, initial conditions in
  the Menck protocol. Keep both names qualified ("parameter-sweep basin study" vs
  "init-perturbation basin stability") until a `lab/analysis` module unifies them.
- **Generalized ARP.** The draft's acceptance criterion — desirable outcomes persist as
  robust basins across all tested *agent* models — extends across the register: a
  defense passes when its outcome basin persists across agent models **and substrate
  forks**. The alpha operationalizes the substrate axis (this register) and the
  resilience axis; the agent-variation axis arrives with the learning delegate
  (backlog). The scorecard's `influence_preserved` point score stays until the basin
  deliverable earns its replacement.
- **Exit time + early-warning signals** (Kramers landscapes, critical slowing down)
  belong in `lab/analysis` when built. The EWS thread and the spectral-margin thread
  are the same mathematics (recovery time diverging as ρ→1) — one thread, eventually.

## 9. Relationship to the alpha plan

The register must not eat A4/A5. **A4 (`influence_exchange`) remains the next build**
— see the 2026-07-24 status note in `docs/abm-suite-design.md` for the A4/A3 design
decisions deposited the same session. Register phasing, interleaved around it:

- **R1** — this doc + the first assumptions card (done with this deposit).
- **R2** — `io_economy`: small (linear algebra, the most JAX-native substrate),
  double-pays by grounding the spectral thread and the analytic instrument rung.
- **R3** — `task_economy`: the rebuild flagship; basin/phase maps in its definition of
  done; `compute_economy`'s relabel lands here, once `labor_dependence` is defined.
- **R4** — cross-substrate scorecard: one defense, three substrates, ARP framing.

Suggested kanban items (Obsidian board, transfer by hand): `io_economy` env,
`task_economy` env, `labor_dependence` relabel, basin-map deliverable, assumptions-card
rollout to `governed_commons`.

## 10. Open questions and non-goals

- **The task_economy self-critique (owed, not dodged):** locating results in an assumed
  task-complementarity parameter would be the same move this doc criticizes in
  `compute_economy`'s σ=2. The defense is twofold and must be honored in the build: the
  dial is the *frontier's speed* (a process, observable in principle) while the
  aggregate elasticity is emergent; and no single-substrate result is headline-worthy —
  cross-substrate invariance (§1) is the claim, or there is no claim. If `task_economy`
  cannot show disempowerment *failing* to occur in some honest parameter region
  (Baumol bottleneck regime), its success regions prove nothing.
- **`labor_dependence`'s precise definition** — elasticity at which horizon, against
  which counterfactual — is open; blocking the relabel, by design.
- Non-goals: learning agents in register substrates (backlog, after the classical
  suite); occupational-mobility networks; real-world calibration (toy models by
  design — the honesty guard in `docs/alpha-context.md` is public copy); any change to
  the frozen `GameSpec` boundary.
