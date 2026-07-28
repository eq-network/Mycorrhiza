# influence_exchange — assumptions card

**What this says a polity is.** A polity is an attention structure: who listens
to whom (one row-stochastic matrix) decides whose opinions weight the
consensus, and political power is nothing over and above that weight.

**Assumptions.**

- Influence dynamics are DeGroot with Friedkin–Johnsen anchoring: opinions pool
  as `x ← (1−λ)·own_signal + λ·(Wx)`. The anchor is load-bearing — with a
  pinned AI reservoir, pure DeGroot's only fixed point is "everyone at the
  reservoir's opinion" however dispersed influence is, and the wisdom readout
  could not discriminate (same role as value_contagion's `recovery`).
- The listening matrix is endogenous by preferential attachment: attention
  drifts at `update_rate` toward attractiveness ∝ influence^γ × amplification
  × cap_scale × engagement. Concentration is organic before any AI appears.
- Influence is tracked in-loop by one power-iteration step per tick — for a
  frozen W it converges to the left eigenvector, which for DeGroot IS each
  node's consensus weight (Golub–Jackson 2010). No offline math.
- AI actors are a frozen reservoir: opinions pinned at `ai_bias`, listening
  rows frozen. Their only asymmetry is scheduled `amplification` of
  attractiveness — algorithmic reach, not persuasive content (content lives in
  value_contagion; this substrate deliberately measures structure only).
- The initial listening graph is an Erdős–Rényi draw: heterogeneous degrees
  seed the heterogeneous eigenvector that preferential attachment amplifies (a
  uniform start is a symmetric fixed point — nothing would concentrate).
- Absent: parties/elections/representation, a state actor, media beyond the
  network, strategic messaging, hierarchy (concentration emerges on a flat
  network — the abm-suite design choice), agent learning.

**Classical result reproduced.** Golub–Jackson consensus weights: with
`update_rate=0` the in-loop influence vector lands on the left eigenvector of
W (L1 < 1e-3), and the wisdom-of-crowds condition breaks measurably under
amplification (`tests/test_substrate.py::test_fixed_w_influence_converges_to_left_eigenvector`,
`::test_wisdom_dispersed_vs_amplified`).

**The dial.** `amplification` (with `amp_onset`): 1.0 = organic preferential
attachment (concentration but human-held, share ≈ 0.87); 4.0 from t=50 =
capture (human share ≈ 0.54 at T=400 and falling; consensus error tripled).

**Instrument.** Mainline indicators: `human_influence_share`,
`influence_gini`, `centralization` (metrics/families/concentration.py),
`consensus_error`. Defenses: the `political` mechanism family — `sortition`
(civic lottery over attention, scheduled) + `influence_cap` (attractiveness
damping above a share cap); calibrated 2026-07-27 so the four conditions
separate: organic 0.87 / amplified 0.54 / defended 0.74 (cap 0.04, share 0.5,
cadence 15, T=400, 6 seeds).

**Lineage.** Originates here (A4, alpha plan). Boundary and reservoir idioms
from value_contagion; scheduled-threat idiom from compute_economy's arrival.
`attract_boost` is a declared no-op seam for A5's `persuasion_shifts_politics`
coupling.

**Status.** Live v0 (2026-07-27, this deposit): 8-rung ladder passing. Not yet
in `experiments/benchmark/scenarios.py` (needs a ScenarioSpec + influence-NOW
instrument decision); public /lab status chip stays "in-design" until that
lands and someone flips it deliberately.
