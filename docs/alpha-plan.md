# Alpha plan — from one live scenario to a benchmark suite

*Companion to `docs/alpha-context.md` (the why). This file maps the five Lab
scenarios onto this repo's actual extension points and phases the work. It is
inspiration + shape, not a sprint plan — reorder freely; keep the definitions
of done.*

## What "alpha" means

An external researcher can:

1. `pip install -e .` and run **scenario 1 end-to-end** — undefended baseline
   vs ≥2 defense mechanisms — from one example script;
2. read a standardized **scorecard** (influence preserved + per-scenario
   measures) out of the run;
3. **extend one catalog** (add a mechanism or environment) following
   EXTENDING.md and have it appear in the benchmark harness without touching
   core;
4. see at least **one disempowerment environment** (economic) running beyond
   the commons.

Modularity is the acceptance test: every piece below should land as a catalog
entry, a paradigm, or an experiment — never as new core abstraction. (Linux
inspiration: grow the kernel's *ecosystem*, not the kernel.)

## Scenario → engine mapping

| Scenario | Environment | Mechanisms (defenses) | Metrics | Exists today |
|---|---|---|---|---|
| 1 Commons | `environments/governed_commons` (**live**, built fresh 2026-07-13 — `resource_game` was orphaned pre-`src/` code, not hardened) + `agents/ai_delegate` | `quota_vote`, `graduated_sanction` (**live**, `mechanisms/democracy.py`); monitoring, polycentric follow | stock %, harvest Gini, compliance, `influence_fidelity` | `environments/governed_commons`, `experiments/benchmark_commons`, `examples/04` (fishing_commons was a stub; governed_harvest self-deprecated) |
| 2 Economic | `environments/compute_economy` (**live** 2026-07-14: CES production, scheduled AI arrivals via activation masks, classical rule-based agents) | `ai_revenue_tax` + `ownership_cap` (**live**, `mechanisms/fiscal.py`); participation subsidy follows | output, wage, labor share, human income share, income Gini/HHI | `experiments/benchmark/scenarios.py`, validation ladder in `compute_economy/tests/` |
| 3 Cultural | NEW `value_epidemic`: SI/SIS-style diffusion on trust adjacency; persuader agents w/ ramping persuasive power | provenance, human-weighted curation, understandability constraint | AI-origin share, variant fidelity, diffusion rate | — (adj_matrices + transforms suffice) |
| 4 Political | NEW `influence_exchange`: influence as an evolving edge/attr; scheduled amplification of few nodes | delegative democracy / AI delegates, sortition, dependence-preserving revenue | influence Gini/HHI, centralization, responsiveness lag | polycentric paradigm adjacent |
| 5 Combined | Composition, not a new env: three domain layers in one `GraphState` (`adj_matrices` is already plural) + cross-domain coupling transforms | portfolios of the above | coupling strength, correlated decline, **defense transfer gap** | compile_pipeline handles ordering |

Notes on mechanics:

- **Actor arrivals** (economic): `GraphState` arrays are fixed-size under
  `lax.scan` — model arrivals as pre-allocated nodes with an activation mask
  flipped by a scheduled transform, not dynamic allocation.
- **Schedules as experimental dials** are first-class already
  (`core/schedule.py`); "schedules set at the start that change the underlying
  dynamics" = `ScheduleEntry` cadences + scheduled parameter-ramp transforms.
- **Ramping persuasion / amplification**: parameter ramps are transforms
  writing an evolving attr — keep static config closed-over (per CLAUDE.md),
  only the ramp value lives in state.
- **Coupling transforms** (combined): small, named, individually testable
  morphisms (e.g. `economic_power_buys_persuasion`) with declared
  reads/writes so `compile_pipeline` orders them.

## The benchmark harness (the alpha-defining piece)

A thin layer in `experiments/` (promote later only if the lab razor passes):

- **Run spec**: (environment, defense pipeline, schedule, seeds, T) — one dict.
- **Scorecard**: influence-preserved score (0–1 vs undefended baseline
  trajectory) + the scenario's named measures; JSON out, so the eq-network
  leaderboard can consume real rows later.
- **Baseline discipline**: every environment ships its undefended baseline as
  a named config — the page's animations are illustrations of these baselines
  and should eventually be *generated from* them (deferred: real-run export →
  page).
- Reproducibility: seeds explicit, `run_scan_batch` over seeds, bootstrap CIs
  via `lab/analysis`.

## Metrics catalog additions

- `influence_preserved` (the leaderboard score) — needs a definition per
  environment; start simple (share of welfare/decision-weight attributable to
  human nodes vs baseline), document the choice, iterate.
- `gini`, `hhi`, `centralization` — generic, reusable in-loop readouts.
- `spectral_margin` (1−ρ of the linearized influence operator, power-iteration
  per tick — JAX-friendly) — **research thread, not an alpha promise**;
  verification items in alpha-context.md §diagnostics.

## Phasing (each phase independently shippable)

- **A0 — Commons vertical slice.** ✅ **Done 2026-07-13** (same-day upgrade:
  MDP boundary + counterfactual influence — see `docs/game-boundary-design.md`).
  `governed_commons` is an open `GameSpec` (observe/step boundary) closed with
  the `ai_delegate` policy; `quota_vote`/`graduated_sanction` mechanisms
  (generic `vote`/`policy_target` contract); **influence is measured causally**
  (`environments/counterfactual.py`: paired same-key rollouts, finite-Δ
  ask-shifts). `python examples/04_benchmark_commons.py` reproduces the
  headline: baseline 0.00 / quota-only **−0.07** / quota+sanctions **0.91**
  causal influence preserved (stock 0.00 / 0.50 / 0.79). First finding: the
  correlational fidelity score (0.49/0.63) hides that unenforced quota voting
  restores almost no causal influence — enforcement is what makes the vote
  channel govern (details: `experiments/benchmark/README.md`).
- **A1 — Benchmark harness.** ✅ **Done 2026-07-13; generalized multi-scenario
  2026-07-14** (`experiments/benchmark/`): `RunSpec` + per-scenario
  `ScenarioSpec` registry + scorecard JSON with bootstrap CIs; conditions are
  **(mechanism, config, ScheduleSpec) triples** — mechanisms are pure rules,
  the schedule owns timing (`core/schedule.py::scheduled`). Adding a defense =
  one `mechanisms/REGISTRY` line + one triple.
- **A2 — `compute_economy`.** ✅ **Done 2026-07-14** (classical-ABM reframe:
  `docs/abm-suite-design.md`). CES production with the substitutability
  assumption as the disempowerment dial; validation ladder passes
  (Cobb-Douglas share ≡ α exactly, no-AI steady state, reinvestment
  concentration); the §2 dependence-decay curve reproduces (labor share
  1.0 → 0.15) and fiscal defenses bend it (0.33 → 0.63 with tax + ownership
  cap). Headline finding: **influence-from-birth ≈ 1.0 in every regime while
  influence-NOW (one-shot mid-run counterfactual) collapses undefended
  (0.34 vs 0.79–0.83 defended) — the historical/current influence gap is
  gradual disempowerment operationalized.**
- **A3 — `value_epidemic`.** Trust-network diffusion, embedded persuaders,
  ramping persuasive power; provenance/curation defenses. *Done when: AI-origin
  share trajectory + one defense comparison runs.*
- **A4 — `influence_exchange`.** Concentration metrics + scheduled
  amplification; delegative/sortition defenses. *Done when: the concentration
  curve is measurable and defense-sensitive.*
- **A5 — Coupling + the flagship experiment.** Cross-domain transforms; run
  every A0–A4 winning defense in the coupled system; compute **defense
  transfer gaps**. *Done when: the per-domain vs coupled scorecard exists —
  this is the result the whole pitch stands on.*
- **Parallel research thread (no gate):** spectral-margin formalization + EWS
  false-positive literature; only enters the public page after Stage 3
  (literature-grounded).

## Explicit non-goals for alpha

- The visual interface (graph editor / schedule UI / dashboard) — sketches
  only; revisit after A5.
- Interactive web demos (composition loop, poke-the-system) — page v3 ideas,
  blocked on the harness producing real trajectories to serve.
- LLM/eager-tier agents in benchmark scenarios (pure tier only — keep runs
  vmap-able and cheap).
- Real-world calibration of any environment (toy models by design; the honesty
  guard is public copy).
- Web leaderboard backend — scorecard JSON is enough until there are external
  submissions.

## Sync obligations

When a scenario goes live here: flip its status chip + swap illustrative
leaderboard rows in `eq-network/src/content/lab.ts` (the page is the public
claim surface for this repo), and consider relisting the page in the site nav
(currently unlisted by choice, 2026-07-13).
