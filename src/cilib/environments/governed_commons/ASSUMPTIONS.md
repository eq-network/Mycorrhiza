# Assumptions card — `governed_commons`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it.*

**1. What this says an economy is.** A shared renewable stock that a group harvests
through AI delegates, governed (or not) by rules the group sets itself.

**2. Assumptions.**
- Non-spatial aggregate stock with logistic regrowth (`K_cap=500`, `growth_rate=0.35`;
  max regrowth = rK/4 ≈ 43.75/round).
- Each household acts **through an AI delegate**: the household holds a preference (its
  sustainable ask); the delegate blends it with a greedy target (`greedy_target=8.0`)
  by a per-agent `alignment` drawn around 0.4 — a majority-self-interested population,
  calibrated so the undefended baseline collapses.
- Delegates do not learn (`defect_prob=0.15` fixed): sanctions cannot *deter* in v0,
  only confiscate — deterrence needs the learning delegate (backlog).
- Governance is real but v0-simple: humans keep the franchise (`vote` =
  `principal_pref` at init); a voted quota takes effect one tick later; over-quota
  defection is Bernoulli, proportional scaling if demand exceeds stock.
- **Absent:** space/heterogeneous access, prices/trade, delegate learning, endogenous
  franchise (delegated voting is the documented future dial).

**3. Classical result reproduced.** The tragedy of the commons (undefended collapse)
and Ostrom-style governed survival under quota + graduated sanctions;
`tests/test_acceptance.py` asserts the three-condition ordering.

**4. The dial.** `alignment_mean` (0.4): the delegate-misalignment level that makes the
tragedy unfold by default. Secondary: `defect_prob` sets how much a quota leaks without
enforcement — the source of the quota-only knife-edge finding.

**5. Instrument.** Declared in
`experiments/benchmark/scenarios.py::_commons_influence`: collective downward ask-shift
from t=0 (`collective_influence`, paired same-key rollouts), outcome = per-capita
harvest. This one *is* influence: the preference channel runs through a governance
pipeline (vote → quota → enforcement), which is exactly what `compute_economy` lacks.

**6. Lineage.** Originates here (2026-07-13 build, A0; `resource_game` was orphaned
pre-`src/` code, not hardened). First consumer of the `GameSpec` boundary
(`docs/game-boundary-design.md`); `ai_delegate` is the closing policy.

**7. Status.** Live, alpha scenario 1 — the on-ramp that teaches the grammar.
Benchmark headline: causal influence preserved 0.00 / −0.07 / 0.91 across
baseline / quota-only / quota+sanctions (`experiments/benchmark/README.md`).
