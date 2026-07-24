# Assumptions card — `io_economy`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it.*

**1. What this says an economy is.** A network of recipes: every product needs
ingredients from other industries plus labor, and at the end people buy finished goods.

**2. Assumptions.**
- Leontief technology — fixed coefficients, **zero substitutability anywhere** (σ=0,
  the register's maximal-complementarity bracket; the world most favorable to human
  indispensability).
- Prices fixed at 1 (quantity-side IO); wages are the only ordinary value added
  (zero-margin sectors, l₀ = 1 − column sum); the AI-cognition sector's value added is
  a margin it owns.
- Output follows orders: one Neumann step per tick (`x ← Ax + d`) — the classical
  Leontief solution is the fixed point, convergence ⟺ ρ(A) < 1.
- AI enters by **editing recipes**: after `sub_onset`, ordinary sectors replace labor
  with purchased AI cognition at cost parity (exogenous schedule — endogenous adoption
  is `task_economy`'s job).
- Households spend all of last round's income by fixed preference weights; equal wage
  split; no saving, no prices to respond to.
- `reinvest_rate` of the AI margin becomes self-directed AI demand; the rest is
  hoarded (a demand leak — load-bearing, see field 4).
- **Absent:** price adjustment/substitution responses (by design — that's the
  bracket), savings/credit, governance channel (same honest limitation as
  `compute_economy`: defenses are experimenter-imposed), heterogeneous sector
  topology (uniform block; `networks.py` generators are the future dial).

**3. Classical result reproduced.** The Leontief fixed point exactly (rebalancing is
stationary at `(I−A)⁻¹d`); attribution ≡ 1 with no AI demand; **hypothetical
extraction ≡ the analytic inverse** — the counterfactual instrument's move recovers
`1ᵀ(I−A)⁻¹Δd_H` in closed form (`tests/test_validation_ladder.py`), giving
`environments/counterfactual.py` its first analytic anchor.

**4. The dial.** `sub_rate`/`sub_onset` drive the twist; **`reinvest_rate` selects
between Gradual Disempowerment §2's two endpoints** (measured, T=250): at 0.3 the
hoarded margin drains demand and everything winds down (absolute: wage bill 16 → ~0);
at 1.0 demand is conserved and total activity *grows* 21.3 → 32.7 while the human
share collapses 1.0 → 0.01 with unsupervised AI spending ~12.7/tick (relative:
the economy runs for its own loop). ρ(A) climbs 0.25 → ≈0.68 as labor becomes
intermediate flow.

**5. Instrument.** Headline metric: the IO demand-attribution share
`1ᵀ(I−A)⁻¹d_H / 1ᵀ(I−A)⁻¹d` (`metrics.py`), plus `unsupervised_ai_spending` and
`spectral_margin`. Benchmark wiring (channel = spend preferences, outcome =
composition responsiveness) is future work; the defense result is already
mechanism-tested: 50% `ai_revenue_tax` at t=50 holds the human share at ≈0.85 vs ≈0
undefended — the same catalog entry as `compute_economy`, unchanged (cross-substrate
reuse working as designed).

**6. Lineage.** Originates here (2026-07-24, register R2). Design sources: Leontief
IO analysis (demand attribution, hypothetical extraction — Miller & Blair), Farmer-school
sequential rebalancing (production-network dynamics, not equilibrium solving).
Register siblings: `compute_economy` (σ assumed, demoted), `task_economy` (designed).

**7. Status.** Live v0 (2026-07-24): validation ladder passing (6 rungs). Not yet a
benchmark `ScenarioSpec`; topology and endogenous adoption deferred by design.
