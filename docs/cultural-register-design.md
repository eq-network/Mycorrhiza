# The cultural register — separation, persuasion, and spectral measurement

*Deposited 2026-07-24 from a design session (forest walk + two source documents). Direction
set by Jonas: apply the model-register epistemology to the cultural scenario (A3), and
settle whether AI cultural disempowerment is about **separate AI cultures** or **more
persuasive AI messages** — the answer is that these are two orthogonal axes, and the
result is a phase diagram over them. Siblings: `docs/model-register-design.md` (the
register pattern this instantiates), `docs/abm-suite-design.md` (A3's row — this doc
supersedes its sketch), `docs/game-boundary-design.md` (the frozen boundary, and where
this design finally forces `close_multi`), `docs/alpha-plan.md` (phasing — A3 still sits
behind A4).*

**Epistemic status: Stage 1–2.** Forest-walk synthesis plus two unreviewed sources. The
verification list in §11 is not optional — several load-bearing claims (networked neutral
drift, the attribution operator under recombination, whether ρ is settable or only
measurable) are unresolved.

## 1. Sources this build stands on

Two documents, both spectral, and they slot into different layers:

- **"A Spectral Theory of Memetic Propagation"** (Hallgren, 2026-02-27) — the *substrate*.
  Messages carry spectral signatures on a **message graph**; agents hold distributions over
  **action policies**; adoption is exponential in cognitive dissonance. This is the model.
- **"Spectral Signatures of Gradual Disempowerment"** (LessWrong, `erTAgDriWAw3evecP`) —
  the *instrument*. Graph Laplacian on the **agent graph**; three governance monitoring
  metrics (human betweenness across boundaries, spectral-gap ratio, Fiedler partition
  alignment). This is what we measure with, and it generalizes past culture.

The pairing is the design: one paper says how culture moves, the other says how to detect
that humans have stopped steering it.

## 2. The two questions, and why they are two axes

"Will there be separate AI cultures, or will AI messages just be more persuasive?" These
are not competing hypotheses. They are **orthogonal dials**, and the disempowerment claim
lives in their interaction:

**Axis S — separation.** Where AI sits in the agent graph `G_A`. A homophily parameter
`ai_homophily ∈ [0,1]`: at 0 AI agents are uniformly embedded among humans; at 1 they
communicate (almost) only with each other, and "AI culture" is a structurally distinct
region of the graph. This is a **network-generator** parameter (`environments/networks.py`).

**Axis P — persuasive advantage.** How much better AI messages are at being adopted. The
memetic paper forces a useful decomposition — a single "AI is more persuasive" scalar hides
three mechanisms with three different defenses:

| Sub-parameter | What AI does | Formal handle | Defense it invites |
|---|---|---|---|
| `reach` | sends more messages / to more agents | multiplier on `r₀` | rate limits, volume disclosure |
| `frequency_targeting` | crafts messages at the fitness optimum λ\* instead of drawing frequency naively | shifts message center frequency `k̄ₘ` toward `α/β` | understandability requirements, frequency-diverse curation |
| `dissonance_targeting` | personalizes: adapts `Pₘ` toward each receiver's `Pᵢ` to minimize `Δᵢ = D_KL(Pᵢ‖Pₘ)` | per-receiver message construction | provenance, personalization limits |

The first two are **substrate parameters** (attributes on nodes/messages). The third is a
**policy** — an AI agent choosing its message given what it observes. That distinction is
exactly where the frozen boundary cracks (§9).

**The deliverable is the (S, P) phase diagram**, not a score. Four corners:

| | low P | high P |
|---|---|---|
| **low S** (integrated) | *pluralism* — baseline; AI variants are part of a mixed culture | **assimilation** — one shared culture, increasingly AI-authored; the monoculture risk |
| **high S** (separated) | *parallel cultures* — AI culture exists, coexists, does not displace | **displacement** — AI culture is both separate and winning; human low-frequency overlap with the prevailing culture erodes → the polarization phase transition, and human culture becomes information-theoretically expensive to sustain |

Only the bottom-right corner is gradual cultural disempowerment. The other three are
genuinely different worlds, and a model that cannot distinguish them is not measuring
anything. Note the diagonal: *separation without advantage is not disempowerment*, and
*advantage without separation is a different pathology* (homogenization, which §7's metric
must score as a loss).

## 3. The substrate ensemble

Three members, following the economy register's discipline (one classical anchor each, one
honest dial, cards colocated). One deliberate change from the forest-walk draft: the
DeGroot/opinion-averaging member is **dropped**, because A4 (`influence_exchange`) already
owns DeGroot with the Golub–Jackson anchor — duplicating it across scenarios would buy no
robustness. Its slot goes to the spectral model, which is the stronger third answer.

| Entry | "Culture is…" | Classical anchor | What it uniquely shows | Status |
|---|---|---|---|---|
| `value_contagion` | …something you catch | epidemic threshold R₀ = 1; complex-contagion threshold at k ≥ 2 exposures (Centola 2010) | spread as a function of **network topology**; where axis S bites hardest | designed |
| `value_replicator` | …a population of ideas competing for finite attention | **neutral drift** (fixation probability = initial frequency with selection off); replicator fixed points; Price equation as accounting identity | cultural **extinction/displacement**; the native home of the disempowerment thesis | designed |
| `value_spectral` | …messages with frequency content, adopted in proportion to how little they demand | **intermediate-frequency dominance** (equilibrium ρ(λ) ∝ λ^α e^{−βλ}, peak λ\* = α/β); **polarization phase transition** at critical low-frequency overlap τ_c | *why* messages spread; the only member where "more persuasive" is a structured claim rather than a bigger number | designed |

**Honest caveat the register must state.** These three are **nested**, not structurally
independent: the spectral model's fast timescale with a constant adoption rate *is*
contagion, and its decay term `δₘ = δ₀ + Σ αₘₘ' πₘ'` *is* replicator competition. That is a
weaker robustness claim than the economy register's (aggregate CES / Leontief / task frontier
share no such containment). The compensating argument: the three still carry
**independent classical anchors**, so agreement across them is agreement across three
separately-validated regimes of one theory — real, but state it as such and do not
oversell it as three independent theories. This is the cultural register's analog of the
`task_economy` self-critique, and it should live on every card.

## 4. What the spectral substrate actually is

For the future implementer, the model in one screen (paper §2, notation preserved):

- **Dual graphs.** Agent graph `G_A` (who talks to whom — slow, quasi-static) and message
  graph `G_M` (semantic similarity between messages — intermediate). Keeping them separate
  is what makes the model tractable and is the single most important structural choice.
- **Frequency basis.** `L_M = D_M − A_M`; eigenvectors `φ_k` with eigenvalues `λ_k` give a
  frequency basis over semantic space. A message's center frequency is
  `k̄ₘ = Σ_k k|c_{m,k}|² / Σ_k |c_{m,k}|²`. Low frequency = smooth/universal; high = sharp/localized.
- **Beliefs are distributions over action policies.** `Pᵢ(θ)` per agent, `Pₘ(θ)` per message.
  This is what keeps "a value" behaviorally grounded instead of an abstract token.
- **Adoption is exponential in dissonance.** `rᵢ(m) = r₀·e^{−βΔᵢ(m)}`, `Δᵢ(m) = D_KL(Pᵢ‖Pₘ)`.
  β is resistance to belief revision.
- **Prevalence dynamics.** `dπₘ/dt = Σᵢ rᵢ(m)(1−πₘ) − δₘπₘ` with competition in the decay.
- **Fitness is non-monotonic in frequency:** `F(m) = e^{−βΔ̄(m)}·S(m) − δₘ`, where salience
  `S(m) = Σ_{m'} D_KL(Pₘ‖Pₘ')πₘ'`. Low-frequency messages spread but bore; high-frequency
  messages are salient but unadoptable; the optimum is intermediate.
- **Message phylogeny is reticulated.** Messages collide and *synthesize* — offspring with
  two parents (paper Fig. 2). This is not decoration; it is what makes §6's attribution
  metric definable.

## 5. Where AI enters

Cleanly, and at three separate places — which is why the two-axis framing works:

1. **Agent graph position** (axis S): AI nodes with `ai_homophily`-controlled attachment.
2. **Message generation** (axis P, sub-params 1–2): AI-authored messages get an `r₀`
   multiplier and/or a frequency draw concentrated near λ\*. Both are attributes; no policy.
3. **Strategic message design** (axis P, sub-param 3): AI selects `Pₘ` against observed
   `Pᵢ`. A policy — and, per the memetic paper's own §5.2 open question, the least
   understood mechanism.

## 6. The attribution metric — what fraction of prevalent culture is ultimately human

The economy register's headline was demand attribution: trace activity back through
`(I − A)⁻¹` to the human final demand it ultimately serves. The cultural analog is
**genealogical**, and the reticulated phylogeny makes it exact:

    human_cultural_share = Σ_m πₘ · h(m)

where `h(m)` is variant m's human-lineage attribution, propagated along the derivation
matrix `M` (`M[i,j]` = the fraction of variant i derived from parent j, rows summing to 1
over parents). Because synthesis attributes offspring as a **weighted average of parents**,
the propagation stays linear, so `h = (I − M)⁻¹ h₀` in the same Neumann form — the *same
operator algebra* as IO attribution and the diagnostics thread's `T = (I − A)⁻¹`, applied to
a third substantive matrix. The fidelity-decay scalar is the per-step discount along each
genealogical edge.

This answers a much better question than "what share of prevalent values are AI-labeled": it
asks **what the prevailing culture ultimately descends from**, which is the thing that can
quietly go to zero while every individual message still looks human-ish.

## 7. Spectral health — and why a scalar share is not enough

The memetic paper's §5.3 supplies the metric that catches the failure mode a naive
"preserve human culture" score would call a win. Healthy discourse = **strong low-frequency
alignment** (shared foundations, so communication is possible) **plus rich high-frequency
diversity** (varied specifics, so innovation is possible). Two pathologies, not one:

- **Polarization** — low-frequency overlap `O^low_AB` collapses below τ_c; communities can
  no longer afford to talk to each other.
- **Groupthink/monoculture** — high-frequency variance is suppressed; everyone agrees,
  nothing new appears. *A curation defense that "protects human culture" by homogenizing it
  lands here, and must be scored as a failure.*

So the cultural scorecard reports the **frequency distribution ρ(λ)**, not a scalar — plus
human attribution (§6) and the generic spectral metrics (§8).

## 8. The generic part: spectral metrics for every environment

This is the "introduce them as part of the general environments we want to measure" move,
and the answer is yes — the LessWrong post's three governance metrics are **not
culture-specific**. Any environment carrying an agent-to-agent adjacency and typed nodes can
report them, exactly as `gini`/`hhi` are reported today. New catalog entry:

`src/cilib/metrics/families/spectral.py`

| Metric | Definition | Reads as |
|---|---|---|
| `fiedler_partition_alignment` | agreement between the sign pattern of the Fiedler vector `v₂` of `L = D − W` and the human/AI node-type split | **the "separate AI cultures" detector** — and the measurement dual of axis S. High alignment = the graph's primary fault line *is* the human/AI boundary |
| `spectral_gap_ratio` | `λ₂(AI subgraph) / λ₂(human subgraph)` | AI reaching internal consensus faster than humans; collective outcomes start reflecting AI dynamics first |
| `human_bridge_share` | humans' share of cross-boundary mediation | declining = AI-mediated chokepoints between domains |
| `low_frequency_overlap` | `O^low_AB = Σ_{k:λ_k<Λ} ⟨P_A,φ_k⟩⟨P_B,φ_k⟩` | the polarization order parameter; the τ_c crossing is an early-warning signal |
| `spectral_margin` | `1 − ρ(A)` | **promote the existing implementation** out of `io_economy/metrics.py` |

Two implementation notes that will save the next agent a day. **Betweenness centrality is
not JAX-friendly** (all-pairs shortest paths, data-dependent control flow) — use
*current-flow / information centrality* via the pseudoinverse of the Laplacian instead,
which is pure linear algebra, vmap-safe, and arguably the better construct here since
influence diffuses rather than routing along geodesics. **Eigendecomposition is fine**:
`jnp.linalg.eigh` on the symmetric Laplacian traces and vmaps.

Honest scope note: of today's environments, none carries an agent-agent adjacency
(`governed_commons` and `compute_economy` are networkless; `io_economy`'s graph is
sector-to-sector, not agent-to-agent). So these metrics land with A3/A4 and only become
*general* as environments grow trust networks — retrofitting the commons with one is a
natural community fork, not alpha work.

## 9. Where `close_multi` actually bites

Precisely, and the answer permits a long deferral:

| Substrate / feature | Heterogeneity handled by | `close_multi`? |
|---|---|---|
| `value_contagion` — persuaders | node attribute: transmission multiplier | No — masking |
| `value_replicator` — fitter AI variants | variant attribute: fitness | No — masking |
| `value_spectral` + axis-P `reach` | node attribute: `r₀` multiplier | No — masking |
| `value_spectral` + axis-P `frequency_targeting` | message attribute: frequency draw | No — masking |
| `value_spectral` + axis-P `dissonance_targeting` | **a policy**: message chosen against observed receiver beliefs | **Yes** |

The rule that generalizes: **`close_multi` is needed exactly when heterogeneity lives in
*agency* rather than in *parameters*.** Since most of the (S, P) phase diagram is reachable
with masking alone, the interface extension is only forced by the personalization corner —
build order in §10 exploits this, mirroring how `io_economy` preceded `task_economy`.

## 10. The honest dial, and the Baumol analog

**The dial: ρ = correlation between a message's fitness and its fidelity to human interest.**
Not "AI's transmission advantage" — that only ever points one way. ρ > 0: the messages that
spread best are the ones most faithful to human meaning (enrichment). ρ < 0: what spreads is
decoupled from, or corrosive to, what serves humans (disempowerment). Operationally ρ lives
in the *persuader's objective*: optimize fitness alone, or fitness conditioned on faithful
carriage. Defenses (provenance, human-weighted curation, understandability requirements) are
interventions **on ρ**.

**The substrate must be able to show AI improving human culture**, or finding the opposite
proves nothing (the `task_economy` Baumol rung, culturally). Three reachable mechanisms:

1. **Prestige/conformist bias** toward human sources — a human gravity well that can beat a
   raw transmissibility advantage (Boyd & Richerson's transmission biases).
2. **AI as faithful amplifier** — if human meaning-making is the scarce complement and AI
   handles transmission/translation, human variants reach *more* minds and rare traditions
   survive. Human attribution share goes **up**.
3. **AI as bridge-builder** — native to the spectral model and the sharpest of the three:
   effective bridging happens at intermediate frequencies, so an AI that generates
   intermediate-frequency messages can *raise* `O^low_AB` between polarized human
   communities and pull the system back from the polarization transition. AI as the reason
   humans can still talk to each other. Directly measurable: run with and without.

If a build of `value_spectral` cannot produce mechanism 3 in some honest parameter region,
the model is assuming its conclusion and the rung has failed.

## 11. Build plan

Strictly behind A4 (`influence_exchange`) — the priority guard stands, and A4 also delivers
the DeGroot machinery and the first agent-graph consumer of §8's metrics.

- **C0 — spectral metrics family** (`metrics/families/spectral.py`). Independent of A3,
  useful to A4, small. Promote `spectral_margin` out of `io_economy`. *Done when:*
  Fiedler alignment recovers a planted two-block partition; gap ratio is correct on a graph
  with a known dense/sparse split; current-flow centrality matches a NetworkX reference on a
  small fixture. **Do this first — it is the cheapest thing on this page and it de-risks A4.**
- **C1 — `value_replicator`.** The honest home of the thesis; masking suffices. *Done when:*
  neutral drift reproduces (fixation probability ≈ initial frequency with selection off),
  a fitness-advantaged variant fixes, and the §6 attribution metric runs.
- **C2 — `value_contagion`.** Cheap, reuses `environments/networks.py`; the topology story
  and axis S's first home. *Done when:* the epidemic threshold reproduces at k=1, the
  complex-contagion threshold shifts as predicted at k≥2, and `ai_homophily` moves
  `fiedler_partition_alignment` monotonically.
- **C3 — `value_spectral`, non-strategic.** Dual graph, KL-dissonance adoption, axis-P
  sub-params 1–2 as attributes. *Done when:* intermediate-frequency dominance appears in the
  equilibrium ρ(λ), the polarization transition is crossable, and **the bridge-builder rung
  passes** (§10 mechanism 3).
- **C4 — strategic persuaders + `close_multi`.** Only now. Extend the boundary
  (`close_multi(game, policies_by_node_type)` via `lax.switch`, per the frozen doc's own
  escape hatch), add `dissonance_targeting`, and produce the full **(S, P) phase diagram**
  with basin-stability shading per `docs/model-register-design.md` §8.

## 12. Handover — what a future agent needs

**Read first:** `CLAUDE.md`, `docs/model-register-design.md` (the pattern this follows),
this doc, `docs/game-boundary-design.md` (frozen interface + the `close_multi` escape hatch).

**Reuse, do not rebuild:**
- `src/cilib/environments/networks.py` — complete/ring/Erdős–Rényi/Watts–Strogatz
  generators. **Axis S is a new generator here** (typed homophilic attachment), not new
  environment code.
- `src/cilib/environments/game.py` — `GameSpec` / `close` / `validate_reads`.
- `src/cilib/environments/io_economy/` — **the closest structural precedent**: an environment
  whose state carries a matrix in `adj_matrices`, evolves it with a transform, and computes
  a spectral quantity in `metrics.py`. Copy its shape.
- `src/cilib/core/schedule.py` — `scheduled` / `ScheduleSpec`; every ramp (persuasive power,
  AI arrival) is a schedule, never bespoke timing code.
- `src/cilib/environments/counterfactual.py` — the causal instruments, incl.
  `intervention_elasticity` (realized-divisor elasticities; read its docstring before
  writing a new instrument).
- `src/cilib/metrics/families/concentration.py` — the pattern a new metric family follows.

**Conventions that will bite (all in CLAUDE.md, all real):** fixed array shapes under
`lax.scan` — pre-allocate message slots and node slots, mask instead of allocating; no
data-dependent Python control flow inside transforms (`jnp.where` / `lax.cond`); static
config closed over by factories, never in `global_attrs`; any `global_attrs` field a
transform writes must be a `jnp` array from t=0 (pytree treedef stability); declare
`.reads`/`.writes` via `@transform`; behavioral tests assert direction/ordering, not
bit-exact numbers; every new environment ships a colocated `ASSUMPTIONS.md` (7 fields,
contract in `docs/model-register-design.md` §6).

**Specific gotchas for this build:** the message graph is a second, *fixed-size* graph —
pre-allocate M message slots with an activation mask, same pattern as `compute_economy`'s AI
slots; the genealogy matrix is (M, M) and must stay row-stochastic over parents or §6's
attribution stops being a weighted average; betweenness needs the current-flow substitute
(§8); `jnp.linalg.eigh` is fine but eigenvector *signs* are arbitrary — Fiedler alignment
must be sign-invariant (compare partitions, not vectors).

**Do not:** build a DeGroot/opinion substrate here (A4 owns it); touch `GameSpec` before C4;
start A3 before A4 ships; report a scalar "AI cultural share" as a headline (§7).

## 13. Open questions — verify before these harden

1. **Networked neutral drift.** Fixation-probability-equals-initial-frequency is a
   well-mixed result; on a graph the neutral dynamics differ. Check evolutionary graph theory
   (Lieberman–Hauert–Nowak 2005) before asserting C1's anchor — a wrong rung is worse than no rung.
2. **Is ρ settable or only measurable?** §10 assumes the fitness↔fidelity correlation can be
   dialed. It may only be *emergent* from the transmission-bias structure — in which case it
   becomes a readout (like `task_economy`'s `adoption_gap`), which is fine, but the design
   changes.
3. **Does the polarization threshold survive discretization?** τ_c is derived in a continuum;
   with K discrete tasks-worth of message slots and finite agents it may smear into a
   crossover. Check before claiming a phase transition.
4. **Complex contagion's sign is unknown.** Does raising the exposure requirement k protect
   humans (harder for an AI monoculture to cross threshold) or entrench incumbents? Plausibly
   regime-dependent — a phase-diagram question, not a prior.
5. **Both sources are unreviewed.** The memetic paper is a working draft with unresolved
   internal references; the LessWrong post is a blog post. Neither has external validation,
   and the LessWrong post states its own scope limit — the Laplacian/diffusion approximation
   breaks down under strategic positioning, which is *precisely* C4. Treat C4's instrument
   readings as diagnostics, not ground truth.
