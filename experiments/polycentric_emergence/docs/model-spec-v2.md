# Model Spec v2 — Polycentric Commons as Endogenous Causal Emergence

Concrete, implementable spec for the CI-Library experiment. Updates the approved plan
with the design-session insights and the MAS-replication framing. Companion to the paper claims rewrite
(`…/requisite-variety-emergence/claims-v2-polycentric-complexity.md`) and the prior-work map
(`research/polycentric-governance-prior-work.md`).

## What's new vs the approved plan
1. **EI∧fit, not EI alone.** Add a `fit` metric (setpoint-to-local-conditions match). The headline
   is the *(EI, Fit)* gap, not the EI peak.
2. **Heterogeneity is the primary independent variable.** Sub-communities have different local
   optima; sweep the spread.
3. **Policer-ablation is the load-bearing test** (knock out monitoring/sanctioning ⇒ EI∧fit and
   behavior degrade together, within a rollout).
4. **MAS replication target:** reproduce "decentralized/emergent governance sustains the commons,
   top-down imposition does worse" (Ren 2025 / Perolat–Leibo 2017), then re-describe via EI /
   complexity / active inference.
5. **Rigor:** dynamics-derived intervention (not max-entropy); "right scale" anchored in
   sufficiency/lumpability (Rosas 2024); claim epistemic emergence.

---

## A. Environment: `engine/paradigms/polycentric/`

Tragedy-of-commons substrate, reusing `experiments/governed_harvest/` heavily. One **shared**
logistic pool (no spatial patches — that would pre-impose blocks). Stochastic policies (an
interior EI signal needs micro indeterminism). **Strip pre-baked prosociality** so atomized
collapses (the non-triviality gate).

### Heterogeneity (the new IV)
Each agent `i` has a **local condition** `theta_i` (e.g., its cost/benefit curve or its
sustainable harvest level). `heterogeneity = std(theta)`. A single global quota cannot fit all
`theta_i`; the welfare cost of uniformity (Oates) rises with `std(theta)`. Agents fall into
`B` latent sub-communities by `theta` similarity — the institutions that *should* emerge.

### State (`schema.py`), N agents, L harvest levels
```
node_attrs:
  harvest_weights (N,L)        # bandit logits          [reuse governed_harvest]
  theta (N,)                   # local condition / ideal harvest      [NEW: heterogeneity]
  affiliation_logits (N,N)     # learnable delegation row             [NEW: the crux]
  vote_value (N,)  quota (N,)  # per-agent quota (was global scalar)  [NEW]
  local_health (N,)  monitored (N,)                                  [NEW]
  last_harvest (N,)  cumulative_harvest (N,)  rewards (N,)  fit (N,)  [reuse + NEW fit]
adj_matrices:
  affiliation (N,N) = row_softmax(affiliation_logits)                [NEW; replaces all-ones]
global_attrs (dynamic jnp arrays only):
  resource_level ()  rng_key  step (int32)  affiliation_sum (N,N)    # ΣW_t → W̄
```
`PolyConfig` (frozen dataclass, closed over by transforms, NOT in global_attrs):
`governance ∈ {atomized, monocentric, fixed_poly, endogenous}`, `affiliation_init`,
`freeze_affiliation: bool`, `capture: float`, `hub_mask`, `monitor_cost`, `sanction_strength`,
`heterogeneity`, `B` (n sub-communities), `K_carry`, `growth_rate`, `learning_rate`,
`affiliation_lr`, `alpha_local`.

### Transforms (`transforms.py`), pure `state→state`, RNG via `_split_key`
Reuse `resource_dynamics_transform` as-is; fork harvest/reward/learning for per-agent quota +
local sanction; reuse `weighted_aggregate`/`row_normalize` from
`engine/transformations/bottom_up/vectorized_message_passing.py`.

| transform | reads | writes | note |
|---|---|---|---|
| `make_vote(cfg)` | vote/harvest weights, local_health, theta | vote_value | desired quota given local condition |
| `make_local_quota(cfg)` | affiliation, vote_value | quota | `q_i = Σ_j W[i,j]·vote_value_j` (W=I atomized, ones/N monocentric, blocks polycentric) |
| `make_harvest(cfg)` | harvest_weights, quota, resource_level, rng | last_harvest, cumulative_harvest, resource_level, rng | fork; per-agent cap; shared-pool rescale |
| `make_monitor(cfg)` | affiliation, last_harvest, rng | monitored, resource_level | **costly**: paying agents observe neighbours' overshoot; cost debits pool |
| `resource_dynamics_transform` | resource_level, K, growth_rate | resource_level | **reuse as-is** |
| `make_local_health(cfg)` | affiliation, last_harvest, quota, theta, resource_level | local_health | `α·Σ_j W[i,j]·restraint_j + (1-α)·R/K` |
| `make_fit(cfg)` | last_harvest, theta, quota | fit | `-|harvest_i - theta_i|` (negative local regret) — the NEW measure |
| `make_reward(cfg)` | last_harvest, monitored, affiliation, quota | rewards | extraction − local sanction `sanction_strength·Σ_j W[i,j]·monitored_j·overshoot_j` (sanction costs punisher) |
| `make_learning(cfg)` | rewards, last_harvest, local_health | harvest_weights | reuse multiplicative-weights; local (not global) health |
| `make_affiliation_update(cfg)` | affiliation_logits, restraint/theta, hub_mask | affiliation_logits, affiliation | **endogenous step**, gated off if `freeze_affiliation`; `+capture·hub_mask` |
| `make_affiliation_accumulate` | affiliation, affiliation_sum | affiliation_sum | running W̄ |
| `step_counter_transform` | step | step | reuse |

Round order (compile_pipeline topo-sorts): `vote → local_quota → harvest → monitor →
resource_dynamics → local_health → fit → reward → learning → affiliation_update(gated) →
accumulate → step_counter`.

### Governance configs (same env, differ only in W-init / gate / capture / ablation)
| config | W-init | freeze | capture | monitoring |
|---|---|---|---|---|
| atomized | I | yes | 0 | off |
| monocentric | ones/N | yes | 0 | central |
| fixed_poly | block-diagonal (B) | yes | 0 | on |
| **endogenous** | small random | **no** | 0 | on |
| capture sweep | small random | no | c∈[0,.9] | on |
| **policer-ablation** | (any) | — | 0 | **off** (the C6 knockout) |

### Agent (`agents.py`): `PolycentricAgent(cfg).round_fn()` → `(state,t,key)->state`, threads its
own rng via `_split_key`; per-seed randomness from `init_fn`. Compatible with `run_scan` /
`run_scan_batch`. Active-inference reading documented: `quota` = institution's policy prior;
`affiliation` = blanket; frozen-precise central = calcification.

---

## B. Measurement: `engine/analysis/`

- `effective_information.py` — **done** (Phase 1, pinned to reference). EI, det/deg, leak,
  coarse_grain, stationary, partition_to_S.
- `causal_emergence.py` (Phase 3) — offline pipeline, numpy (no scipy/networkx in deps → implement
  agglomerative clustering + greedy-modularity community detection in numpy):
  - **T construction (two tracks):** Track-S = row-normalized W̄ (structural, pins to reference);
    Track-B = coupling from discretized harvest trajectories (behavioral, the real claim).
  - `nested_hierarchy`, `emergent_partition`, `ei_curve`, `interior_max_test`,
    `null_compare` (surrogate band: phase-randomized / block-shuffle), `shapley_ei` (exact ≤~12).
  - **dynamics-derived intervention:** stationary `p` of T (the system's own occupancy), not
    uniform — already how `macro_tpm`/`stationary` are wired.
- `fit.py` (new, tiny) — `fit_score(harvests, theta)`, `welfare`, `per_block_fit`,
  and the **(EI,Fit) pair** + the EI∧fit gap vs heterogeneity.
- Reuse `engine/environments/commons_metrics.py` (survival/efficiency/equality) as viability.

**Estimator gate (Phase 2, `validate_estimator.py` + tests):** Hoel toy chain with known
EI-vs-scale; i.i.d./single-scale nulls at the rollout/bin/state-space sizes used; assert no false
interior peak. Must pass before trusting any commons result.

---

## C. Experiment: `experiments/polycentric_emergence/run_experiment.py`

1. For each governance config × heterogeneity level × seed batch: `run_scan_batch` (vmap seeds),
   trace `resource_level (T,)`, `last_harvest (T,N)`, accumulate `affiliation_sum` → W̄.
2. Offline: estimate T (Track-B headline, Track-S cross-check), find emergent partition, compute
   `(EI, Fit)`, leak, Shapley-EI, viability (survival/efficiency/equality).
3. **Headline figure:** (EI,Fit) for each regime vs heterogeneity → show endogenous co-maintains,
   monocentric trades fit for EI, gap widens with heterogeneity (C3/H3).
4. **MAS replication:** show endogenous sustains the commons, monocentric/atomized do worse
   (survival/cooperation) — matching Ren/Perolat — and that the (EI,Fit) gap *predicts* the
   survival gap (H_repl).
5. **Ablation:** policer-knockout run; EI∧fit and survival drop together within a rollout (C6/H0).
6. **Capture:** sweep c; fit collapses, Shapley-EI concentrates, even where EI persists (H_cap).

---

## D. Build order (cheapest-kill first) & status
- [x] **Phase 1** EI kernel + tests pinned to reference. **done.**
- [ ] **Phase 2** estimator gate (Hoel toy + nulls). *next.*
- [ ] **Phase 3** offline `causal_emergence.py` (+ numpy clustering, surrogate nulls, Shapley);
  reproduce reference (A)(B)(C) as tests.
- [ ] **core/__init__.py** union-resolve (unblocks env). 
- [ ] **Phase 4** env paradigm package + frozen-W controls (monocentric≈PDD sanity; atomized
  collapses); `run_scan`/`run_scan_batch` smoke tests.
- [ ] **Phase 5** endogenous learning earns structure; H1 surrogate-null falsifier.
- [ ] **Phase 6** EI∧fit + heterogeneity sweep (H3) + MAS replication (H_repl).
- [ ] **Phase 7** ablation (H0) + capture (H_cap) + endogeneity (H4); full run + figures.

## E. Risks / decisions deferred
- scipy/networkx vs numpy-only clustering (going numpy-only for now to avoid new deps).
- exact `theta` parameterization (harvest-ideal vs cost-curve) — starting with harvest-ideal.
- whether a genuine 3rd (federation) blanket forms — let the data decide; don't impose 3 scales.
