# Assumptions card — `compute_economy`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it.*

**1. What this says an economy is.** One machine with two input slots — human work and
computer power — that pays each input by how much a little more of it would help.

**2. Assumptions.**
- Aggregate CES production over total labor L and total compute C (`config.py`:
  `rho=0.5` ⇒ elasticity of substitution σ=2; ρ=0 recovers Cobb-Douglas).
- Factor prices are marginal products; income accounting closes exactly
  (wage·L + r·C = Y, the Euler identity).
- AI compute compounds by a **fixed reinvestment rule** (`reinvest_rate=0.3`,
  `depreciation=0.05`) — reinvestment is not a decision anyone makes.
- AI actors arrive on an **exogenous schedule** (pre-allocated node slots, activation
  masks) — adoption is not driven by competition or returns.
- Households work their preference, mildly wage-adjusted (`agents/labor_supply.py`);
  no learning, no optimization.
- **Absent:** a demand side (output is produced, never consumed — "what the economy
  produces, for whom" is undefined); a governance channel (no votes, no rule-making);
  endogenous adoption; a state actor.

**3. Classical result reproduced.** Cobb-Douglas limit holds the labor share at exactly
α every tick; no-AI economy is a stable steady state; reinvestment concentrates income
(rich-get-richer). `compute_economy/tests/test_validation_ladder.py`. Note the ladder
validates the *accounting* at ρ=0 — the disempowerment twist (ρ=0.5) has no rung.

**4. The dial.** `rho`. σ>1 ⟺ the marginal product of compute stays bounded away from
zero as C grows ⟺ the reinvestment loop can self-sustain (growth factor
`1 − δ + s(1−τ)r` > 1); the tax that opens the loop is `τ* = 1 − δ/(s·r∞)`, in closed
form. At ρ=0 the loop always self-limits.

**5. Instrument.** Declared in
`experiments/benchmark/scenarios.py::_economy_labor_dependence` (relabeled + redefined
2026-07-24): the static output elasticity of human labor, d logY / d logL over the 10
ticks after a one-shot work-preference shift at 2T/3, realized shifts from paired
same-key batches. Measures **economic dependence on human labor**, not influence; the
Cobb-Douglas rung (`tests/test_validation_ladder.py`) asserts it recovers α.

**6. Lineage.** Originates here (built 2026-07-14, A2). Register siblings:
`io_economy`, `task_economy` (both designed, not yet built — register doc §4).

**7. Status.** Demoted to pedagogical rung, 2026-07-24: the labor-share collapse
follows from the assumed σ and fixed reinvestment (assumption-in, assumption-out), and
the model has no channel through which humans exercise influence. Kept as the register's
teaching example and for its closed-form threshold. `docs/model-register-design.md` §2.
