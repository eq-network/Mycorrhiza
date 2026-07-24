# The classical-ABM suite — design frame for the alpha

*Deposited 2026-07-14, direction set by Jonas: the alpha is a program of **classical
agent-based models** — rule-based agents only, no RL policies, no LLM agents, "the
stupid thing first." Mainline interpretable economic variables tracked over time; every
substrate validated against known results before its disempowerment twist is trusted;
defenses never baked in but **attached via the scheduler**. This document is the map;
`docs/alpha-plan.md` tracks status; `docs/game-boundary-design.md` records the
interface the suite is built on.*

**Status (2026-07-24):** two same-session updates. **(a) The Economy row is superseded
by the economy model register** (`docs/model-register-design.md`): `compute_economy` is
demoted to a pedagogical rung — its influence number ≈ the labor share by the CES
identity, so it measures *labor dependence*, not influence (relabel decided,
implementation deferred) — and the flagship economic scenario will be rebuilt as
`task_economy`, bracketed by `io_economy` (Leontief, σ=0). **(b) A4/A3 design
decisions:** A4 builds **before** A3 (A4 is pure composition of existing pieces; A3
carries the documented `close_multi` interface risk, better faced with a second data
point behind us). A4's substrate is DeGroot influence dynamics with the
**Golub–Jackson wisdom-of-crowds theorem** as validation anchor and disempowerment
dynamic in one object: consensus weights = eigenvector centrality, scheduled
amplification breaks the wisdom condition, and the concentration metrics read straight
off the weights. Hierarchy is *not* modeled explicitly — concentration emerges from
amplification on a flat network, organisations are heterogeneous high-capacity nodes
(the `compute_economy` mask pattern), and two-level structure arrives via the
representative-democracy mechanism as a swappable rule. A3 is scoped as multi-strain
SI/SIS: variants carry an origin label and a fidelity scalar decaying on transmission;
the AI advantage is a transmission-rate edge; the epidemic-threshold rung must pass
before persuaders ramp.

## The shape of the program

Each ABM module is four things:

1. **A base substrate** with mainline indicators — variables a economist or policy
   reader recognizes (production, labor share, quotas, concentration), evolving over
   time under fixed-rule agents.
2. **A validation ladder rung** — the substrate must reproduce at least one classical
   multi-agent/economic result before we trust anything else it shows.
3. **An undefended disempowerment baseline** — the failure unfolds by default, visible
   in the mainline indicators AND in a causal influence instrument.
4. **A mechanism surface** — defenses are catalog entries (`cilib.mechanisms`) attached
   as **(mechanism, config, schedule) triples**; the composition principle below.

## Module table

| Module | Status | Environment | Mainline indicators | Validation anchor | Disempowerment dynamic |
|---|---|---|---|---|---|
| Commons | **live** | `governed_commons` | stock %, harvest Gini, compliance, fidelity | tragedy of the commons; Ostrom's governed regimes (also `lab/paradigms/polycentric`) | misaligned delegates strip the stock; asks stop governing outcomes |
| Economy | **live** | `compute_economy` | output, wage, labor share, human income share, compute stock, income Gini/HHI | Cobb-Douglas constant labor share (exact, every tick); no-AI steady state; rich-get-richer concentration from reinvestment | σ>1 substitution + compounding AI compute → labor share decays; influence-*now* collapses while influence-from-birth stays ≈1 |
| Value epidemic | A3 | `value_epidemic` | AI-origin share, variant fidelity, diffusion rate | SI/SIS epidemic threshold on the trust network | replication-fit AI variants displace human-origin culture |
| Influence exchange | A4 | `influence_exchange` | influence Gini/HHI, centralization, responsiveness lag | preferential attachment → concentration | scheduled amplification bends the concentration curve one way |
| Combined | A5 (flagship) | composition, not a new env | coupling strength, correlated decline, **defense transfer gap** | — | per-domain recoverable, jointly locked in |

Network topologies for A3/A4 come from `environments/networks.py`
(complete/ring/Erdős–Rényi/Watts–Strogatz — "different base assumptions" as a dial).

## The composition principle: mechanisms attach via the scheduler

Mechanisms are **pure rules** (what a quota vote computes); schedules own **timing**
(when it fires). `core/schedule.py`:

- `scheduled(mechanism, cadence, phase_offset, onset)` — gates any transform on the
  step counter, scan-safe, reads/writes metadata preserved so `compile_pipeline`
  still orders it.
- `ScheduleSpec` — the serializable dial a benchmark condition carries.
- `onset` is the **regime-shift dial**: "taxation switching on at t=100", a shock
  arriving mid-run, a defense deployed late. One-shot interventions are schedules too
  (`cadence > n_steps, phase_offset = onset`) — which is exactly how the influence
  instrument's mid-run counterfactual is expressed.

A benchmark condition is a list of triples — adding a defense is one line
(`experiments/benchmark/scenarios.py`), and *when* it applies is part of the
experimental design, not the mechanism's code.

## Influence measurement (the two-counterfactual finding)

All headline influence numbers are **causal**: paired same-key rollouts with a
perturbed human-preference channel (`environments/counterfactual.py`), never
correlational fidelity alone. The economy build surfaced a distinction that now
organizes the whole suite:

- **Influence-from-birth** (init-state perturbation): would history differ if human
  preferences had always been different? Stays ≈1.0 even in the captured economy —
  early human behavior is upstream of the very structures that later capture it.
- **Influence-now** (scheduled mid-run intervention): if humans change behavior *after*
  structures entrench, do outcomes respond? Collapses in undefended baselines
  (economy: 0.34 vs 0.79–0.83 defended; the commons quota-only knife-edge: −0.07).

**The gap between the two is gradual disempowerment, operationalized.** Each scenario
declares its channel + outcome + counterfactual design in
`experiments/benchmark/scenarios.py`; the instrument (`collective_influence`,
`intervention_response`, `influence_matrix`) is generic.

## Validation ladder (current rungs)

- Commons: undefended tragedy reproduces; governed regimes survive (pytest:
  `test_acceptance_undefended_baseline_collapses`, the three-condition ordering).
- Economy: Cobb-Douglas limit holds the labor share at exactly α every tick; no-AI
  economy is a stable steady state; reinvestment concentrates income (Gibrat-flavored
  rich-get-richer); Euler identity closes income accounting each tick
  (`compute_economy/tests/test_validation_ladder.py`).
- A3 target: SI/SIS threshold behavior on the trust network before persuaders ramp.
- A4 target: preferential-attachment concentration before scheduled amplification.

## Adaptivity / antifragility roadmap (destination, not this round)

1. **Adaptive mechanisms as ordinary catalog entries** — feedback controllers (e.g. a
   tax rate tracking a target labor share) are just mechanisms; nothing in the
   architecture changes. They earn a place once static defenses have baselines to beat.
2. **Antifragility tests as schedules** — a regime shift (`onset`-scheduled shock:
   faster AI growth, a demand collapse) hits mid-run; the question is whether a defense
   calibrated for the old regime keeps preserving influence. The harness can express
   this today; the study design is the work.
3. **Learning agents** (bandit/Q-learning delegates already exist in
   `lab/paradigms`/`experiments`) — makes deterrence measurable and opens
   strategic-robustness testing (quota gaming, vote manipulation). Deliberately
   AFTER the classical suite is complete: mechanisms first proven on dynamics that
   can't adapt around them.

## North star: a unified information-theoretic measure

The per-scenario influence instruments are stepping stones toward one frame:
**empowerment** (Salge & Polani — channel capacity from human choices to future
states; our finite-Δ responsiveness is its exercised, linearized shadow) and the
**spectral margin** 1−ρ(A) of the influence operator (`influence_matrix` already
computes A; the flagship A5 claim is per-domain ρ<1 with coupled ρ>1 — lock-in
invisible to per-domain monitoring). Epistemic status: research thread
(alpha-context.md §diagnostics), Stage 1–2; the benchmark ships main indicators until
that measure survives the literature.

## Interface status

`GameSpec`/`close()` (docs/game-boundary-design.md) is **frozen** as of the second
consumer: `compute_economy` fit without any interface change. One documented friction
stands: `close()` vmaps ONE policy over all agents — heterogeneous populations
(households vs AI actors; A3's persuaders vs members) are handled by masking inside
`step_fn`. When A3 makes that painful, the extension is `close_multi(game,
policies_by_node_type)` via `lax.switch` — don't build it before then.
