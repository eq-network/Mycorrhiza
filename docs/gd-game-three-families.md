# Three families of mechanics — economy, culture, politics

> **ABANDONED AS A GAME DESIGN 2026-08-01.** The game the three families were
> built for was judged not worth continuing. The proposed shape in §4 — three tabs of levers on a
> live engine — is route B, and it is dead; see
> [docs/gd-game-postmortem.md](gd-game-postmortem.md). What survives is
> engine-side and is read as findings about `ledger_society`, not as a plan:
> the probe tables in §2, the finding in §3 that `belief` is instrumented but
> causally disconnected from every scored outcome, and §6's record of what was
> built and of the pure-tier sealing defect. All of it carries the provenance
> that it exists because the game needed it.

*Deposited 2026-07-31. Direction: the design review produced
scoring and presentation fixes but no new mechanics, and the game should
carry three sets of dynamics around the economy, politics and culture.
Siblings: docs/gd-game-design.md (the card game, built),
docs/gd-game-dynamics-review.md (the diagnosis and the journey-scoring
decision), docs/remote-engine-design.md (the live-lever path, R0 built).
All probe numbers here: harbor dials, wait path, 6 seeds, one field changed
at a time, journey-window metrics, deltas against the unchanged baseline.
Medians/points, no CIs — ordering and magnitude-class claims only.*

## 1. The three verbs the model already has

The three ledgers have different physics, so three families should play
differently rather than deal three flavours of one card:

| family | the model's own mechanism | the verb | how it behaves |
|---|---|---|---|
| economy | the per-agent `allocation` vector over consume / invest / broadcast / lobby / save, spent every tick | **allocate** | zero-sum across five columns, compounding through capital |
| culture | the preferential-attachment kernel that rewires `listening` | **shape** | you cannot move attention, only the parameters governing how it moves; acts with a lag |
| politics | `enforcement`, a bounded stock eroded by funded pressure and refilled only by repair | **spend** | a ratchet with a maintenance floor |

**The coupling is already honest and needs nothing invented.** In this model
money is the only thing that acts on the other two domains: `broadcast_spend`
is what buys reach, `lobby_spend` is what moves the regime, and both are
columns of the same allocation vector. The economy is not one of three peers
— it is the budget the other two are drawn from. That is where "many simple
choices combine into complexity" actually comes from here: five columns that
must sum to one, feeding two downstream domains with different time
constants.

## 2. Probe: which levers actually bite

Baseline wait path: wealth 0.4492, attention 0.4874, composite 0.4569,
enforcement 0.5968, flip at tick 26.

**Economy — dominant, by an order of magnitude.**

| lever | Δ wealth | Δ composite | Δ enforcement | flip |
|---|---|---|---|---|
| invest 0.05 → 0.25 | **+0.539** | **+0.367** | +0.403 | **52** |
| save 0.20 → 0.40 | +0.331 | +0.058 | +0.219 | 28 |
| lobby 0.02 → 0.15 | +0.248 | +0.094 | +0.403 | 26 |
| broadcast 0.03 → 0.20 | +0.009 | +0.014 | +0.007 | 38 |
| wealth_spend_rate 0.15 → 0.30 | −0.055 | −0.018 | −0.074 | 26 |

**Culture — weak, and one lever is inert.**

| lever | Δ wealth | Δ composite | flip |
|---|---|---|---|
| gamma_w 1.0 → 0.6 (flatter attention) | +0.037 | +0.023 | 26 |
| gamma_w 1.0 → 1.4 (winner-take-all) | −0.017 | −0.010 | 27 |
| update_rate_w 0.08 → 0.03 (slower-turning town) | +0.011 | +0.013 | **36** |
| self_weight_w 0.15 → 0.40 | +0.000 | +0.000 | 26 |
| susceptibility 0.7 → 0.4 | **0.0000** | **0.0000** | 26 |

**Politics — moderate, and three levers are near-duplicates.**

| lever | Δ wealth | Δ composite | Δ enforcement | flip |
|---|---|---|---|---|
| repair_rate 0.02 → 0.04 | +0.132 | +0.046 | **+0.206** | 26 |
| self_weight_d 0.30 → 0.55 | +0.088 | +0.057 | +0.059 | 26 |
| churn 0.05 → 0.15 | +0.088 | +0.056 | +0.059 | 26 |
| attention_to_ballots 2.0 → 0.5 | +0.084 | +0.048 | +0.056 | 26 |
| entrenchment_gain 0 → 0.02 | −0.007 | −0.002 | −0.011 | 26 |

## 3. What the probe changes about the plan

**The game's deepest mechanic is the one it never exposed.** Household
investment moves the composite by +0.367. The entire existing five-card deck
spans 0.057 across all thirteen harbor paths. The most consequential decision
available in this society — what fraction of its income the town puts into
building its own productive capacity rather than consuming it — has never
been on a card. It is also the only lever probed that substantially delays
the flip, from tick 26 to 52, because the newcomers' advantage is capital
accumulation and investment is the one move that competes with it directly.

**The families are not symmetric, and the design should not pretend they
are.** Economy is upstream and powerful; culture is slow and weak but is the
only family that touches when the town turns; politics is a ratchet you can
defend but not reverse. Balancing them to equal authority would be tuning the
model to a game-design wish. Designing around the asymmetry is the honest
move and the more interesting one: the economy is the engine, culture is the
lead-time game, politics is the holding action.

**Three politics levers are three doors to one room.** Churn, franchise floor
and the culture→politics dial land within 0.005 of each other on every
readout, which suggests they act through the same path — keeping the human
ballot share up, raising the tax, returning wealth. As game choices they are
not three choices. Politics needs `repair_rate` and `entrenchment_gain` as
its distinctive pair, and at most one of the ballot-shape three.

**The belief field is a sink — confirmed, not conjectured.** `susceptibility`
changed nothing to four decimal places on every ledger metric, and reading
`dynamics.py` says why: `belief` is written by `pool_belief`, read by
`pool_belief` and by the trace, and by nothing else. No ledger, no
attractiveness, no policy target reads it. So the whole persuasion story —
who believes what, and how strongly they anchor on their own signal — is
currently instrumented but causally disconnected from every scored outcome.
The `belief_capture` metric does move; the ledger shares cannot.

This is an engine finding before it is a game finding, and it is the single
most consequential thing in this probe for the library rather than the game.
A cultural family built on beliefs would be a family of controls that change
a number nobody is scored on. Either belief gets a path into the ledgers —
the obvious candidate is `ideal`, so that what people come to believe moves
what they vote for — or culture in this model means the attention kernel's
shape and nothing else. That is a modelling decision with WP2 implications
and it is not mine to make. `self_weight_w` was also nearly inert and is a
separate, smaller question about the reallocation kernel's floor.

**`entrenchment_gain` is the only clean downside lever found.** It is off by
default and the config calls that "the honest region (WP3)". Turning it on is
a claims-bearing model decision, not a difficulty setting, and would need its
own dated entry — but it is exactly what the design review asked for when it
asked for a way for a path to score below waiting.

## 4. Proposed shape (design conjecture)

Three tabs, one shared budget, on the live engine — the branch tree cannot
carry continuous levers over time (docs/remote-engine-design.md §1).

- **Economy — allocate.** The five-column vector as the primary control,
  constrained to sum to one, adjustable over time. Consumption is not a dead
  column: it is the town's standard of living and should be scored, or
  starving the town becomes a free strategy.
- **Culture — shape.** `gamma_w` and `update_rate_w`, both acting with a lag,
  plus the existing reach cap. This family's whole lesson is lead time: its
  levers are near-worthless once the flip has happened, which the probe
  already shows.
- **Politics — spend.** `repair_rate` against a per-tick enforcement upkeep,
  with `entrenchment_gain` as the standing hazard if it is turned on. One
  ballot-shape lever, not three.

Costs stay in-model per the existing convention: political intensity draws
enforcement upkeep per tick (`PolicyLeverConfig.upkeep`, built), cultural and
economic moves are paid out of the allocation vector itself, which is
conserved by construction.

## 5. What a critic should press on

Every number here is one field moved from harbor's dials on the wait path at
6 seeds with no CIs; interactions are unprobed and the families will not
compose additively — the economy funds the other two, so joint sweeps are
required before any of this is priced. The magnitudes are large enough that
the ordering claims are probably safe and the exact values certainly are not.
Exposing household allocation makes the current five-card deck numerically
irrelevant, so adopting this obsoletes the shipped prices and the committed
harbor expectations, which is a declared revision and not a retune. The
consumption column is the sharpest hazard: if the score counts only the three
ledger shares, the optimal play is to starve the town into investment, and
the game would be teaching a lesson nobody intends. And the whole proposal
moves the game onto a live server, which is a deployment and an attack
surface the static site does not currently have.

---

## 6. Built — 2026-07-31

*What exists in the engine as of this date. Everything below is reproducible
from `(env, config, seed)` in this repo; nothing here is on the web side.*

### What exists now

**Six per-tick substrate ports.** `ledger_society/state.py` promotes
`gamma_w`, `update_rate_w`, `churn`, `repair_rate` and `entrenchment_gain` to
`global_attrs` seeded from their config fields, next to the pre-existing
`reach_cut_now`; `rewire_listening`, `rewire_delegation` and `update_regime`
read them per tick. This is plumbing, not a model change: an untouched run is
the closed-over model element-for-element, verified across the change boundary
and at eight off-default configurations.

**Three lever families**, `mechanisms/families/{economy,culture,politics}.py`,
registered as `economy_levers` / `culture_levers` / `politics_levers`. Each is
one composed transform reading one `(T, P)` plan array carried in `global_attrs`
as a dynamic pytree child, so plan values are data — one compiled program serves
every plan, and plans `vmap` alongside seeds. Levers, ranges and typing:
`mechanisms/families/README.md`.

**The write map**, checked as a test:

    economy   alloc_pref, wealth
    culture   gamma_w_now, update_rate_w_now, reach_cut_now
    politics  wealth, intervention_spend, enforcement, delegation,
              repair_rate_now, entrenchment_gain_now

Culture is disjoint from both siblings. Economy and politics share `wealth`, and
that is the design's third failure mode expressed as a data hazard rather than
as prose: funding the office competes with investing the hoard. `enforcement`
keeps exactly one writer; the siblings' political acts are billed to it as data
through the politics plan's `external_intensity` column, priced by
`external_intensity_of` off their own plan rows.

### Composed-probe numbers

All three families in one `ledger_society` pipeline, config defaults, under
`lax.scan`, 60 ticks. Ordering claims only — no CIs, and the substrate is the
config default rather than section 2's harbor dials, so these numbers are not
comparable to the probe table above.

- **Neutral bundle seals.** All three transforms attached with all-zero plans,
  4 seeds by 60 steps: **128 of 128 arrays bit-identical** to a plain run with
  no mechanisms and no plan arrays — every node attribute, both adjacency
  ledgers, every global, every traced series. The only new state is the three
  plan arrays; nothing is dropped.
- **Each family alone moves its headline quantity**, median over 4 seeds,
  terminal value. Economy `d_invest` +0.15 takes human capital 30.37 to 287.98.
  Culture `gamma_w_delta` makes the HHI of `listen_influence` monotone: 0.0947
  at −0.4, 0.1425 at neutral, 0.1782 at +0.4. Politics `repair_spend_rate` 0.05
  takes enforcement 0.9563 to 0.9634 and human wealth 40.48 to 27.05 — the lever
  and its price in the same run.
- **The three compose engaged.** A levy of 0.20 plus a flatter, slower,
  reach-capped culture row prices at `external_intensity` 2.681; running that
  with the office funded and seats by lot for 60 ticks moves capital 30.69 to
  258.66, wealth 41.03 to 0.37, enforcement 0.959 to 0.787, attention HHI 0.149
  to 0.041. Both ledgers stay row-stochastic to 2.4e-07, `alloc_pref` rows sum
  to 1.000000, enforcement stays inside [0, 1]. The bundle traces: `jit` plus
  `vmap` over 4 init seeds by 2 plans from one compiled program.

### One defect found and fixed during integration

The neutral politics plan was **not** bit-identical under `lax.scan`, only
eagerly — about 1 ULP on `delegation` at tick 1, propagating into `influence`
and `position`. The cause is not lever arithmetic. A literal no-op writing
`diag + (D - diag)` back to `delegation` reproduces the identical 1.49e-08
drift, because under a fused tick XLA re-fuses the producer of `D` and does not
round its two references alike. The sortition write is now
`D + sort_r * delta`, whose neutrality is structural — a zero slider multiplies
the whole perturbation by exactly 0 regardless of rounding inside it — and the
scan-tier rung is asserted as a test. **`make_policy_levers` still carries the
old form.** Its own test already names the weaker guarantee, "neutral to the
ulp", and it was left alone rather than retuned. The general lesson is a library
one: sealing conventions have to be pinned in the pure tier, because the eager
tier does not fuse across the mechanism slot and cannot see this class of
failure.

### What each family can and cannot do

- **Economy** steers `alloc_pref`, not `allocation` — the policy's action
  channel is overwritten at the head of every tick, so a mechanism-slot write to
  `allocation` is dead. The consequence is deliberately not hidden: a change
  lands **the following tick**. The spec's `wealth_spend_rate` drawdown lever is
  **not built**; it needs a `drawdown_now` global in `state.py` and a term in
  `make_allocate`, so the deposit's one negative-signed economy lever is not
  exposed. Standard-of-living scoring is not here and should not be:
  `consume_spend` remains a terminal sink, so starving the town is legible but
  does not bite inside the simulation.
- **Culture** shapes the attention kernel and nothing else. It does **not**
  write the broadcast column of `alloc_pref` — that is economy's conserved
  vector — and it does **not** charge enforcement upkeep, because `enforcement`
  has one writer. Its cost model ships as the pure function `culture_upkeep`,
  which politics calls; the cost split crossing the brief is a ruling still open.
  Section 3's finding stands unchanged: `belief` is instrumented and
  causally disconnected from every scored outcome, so culture in this model
  means the attention kernel's shape and nothing else until that is decided.
- **Politics** owns `enforcement` and pays the whole town's political bill. It
  ships two live sliders; `repair_rate` and `entrenchment_gain` are present as
  scenario dials at 0 = hold, because a free dial is not a decision and because
  arming entrenchment is a claim about the world that owes its own dated
  ASSUMPTIONS entry. `self_weight_d` stays a pre-run constitutional constant —
  there is no `self_weight_d_now` port. `enforcement_rest` is arithmetic off the
  `update_regime` fixed point, labelled as such, not a measurement.

### What remains

Out of this scope by instruction: the live endpoint wiring and the game UI.

1. **The plan keys are not in the state schema.** `make_state` creates none of
   `economy_plan` / `culture_plan` / `politics_plan`, so `build_game`'s
   `validate_reads` rejects all three families; composition today goes through
   `build_step_fn` plus the `attach_*_plan` helpers. The clean fix mirrors
   `policy_horizon`: three horizon fields defaulting to 0, three conditional
   keys, and `build_game` appending the transforms. That is a state-schema
   change which widens the system-graph artifact and wants its own intent.
2. **`drawdown_now`** for the economy family's sixth column, per above.
3. **A `ledger_society` metric for attention concentration.** The culture tab
   has no instrument of its own. `hhi_of` over `listen_influence` is proven
   computable from a traced field and needs to land in `metrics.py`.
4. **Joint sweeps.** Section 5's warning is undischarged: every number in
   section 2 is one field at a time, the families do not compose additively
   because the economy funds the other two, and the probe above is a smoke test
   at 4 seeds with no CIs, not a pricing run. Nothing here prices a lever. The
   politics spec's pre-registered acceptance tests — three doors, downside, wait
   path in the middle third — and the `e*` sweep over intensity are still unrun.
5. **A late-lever measurement for culture.** Every culture effect on record was
   measured with the field changed from `t = 0`, so "near-worthless after the
   flip" remains a design conjecture. The plan array can now express a late
   lever, which is exactly what makes the measurement possible.

### What a critic should press on, for this section

The bit-identity result is the strongest claim here and it is a *null* result:
it says the levers are inert when neutral, not that they are right when engaged.
The direction checks are single-lever, 4 seeds, no CIs, on the default config
rather than the harbor dials section 2 used, so they cannot be compared to the
table above and none of them is a magnitude. The engaged-together run is one
seed and one plan chosen to be legible, not a strategy anyone would play; that
wealth falls to 0.37 while capital reaches 258 is the invest-and-starve corner
section 5 already flagged as the sharpest hazard, reproduced here rather than
solved. And the defect found during integration was found by a probe nobody had
asked for, which is evidence that the per-family test suites — all green — were
testing the wrong tier, and there is no reason to assume this was the only place
that was true.
