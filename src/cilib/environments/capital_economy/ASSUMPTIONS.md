# Assumptions card — `capital_economy`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it.*

**1. What this says an economy is.** A network of recipes in which automation has
an owner: AI systems hold capital that earns the automation slice of sector value
added and must pay upkeep out of revenue before it can compound.

**2. Assumptions.**
- Leontief technology (σ=0 bracket inherited from `io_economy`): fixed hub+chain
  coefficients, prices pinned at 1, output follows orders one Neumann step per tick.
- **Owned capital with upkeep-before-profit**: automation share `a_j = eK_j/(eK_j+1)`
  (saturating by construction — the income-share floor is functional form, not a
  finding); revenue pro rata by stock; upkeep settled `min(rev, mK)` — the unpaid
  remainder is *physical* charge-to-stock, never a money flow; profit reinvests at
  `s`, hoards drain at `c_A`, losses eat the stock.
- **Money conserved at every closure** (checked every test run): the invariant
  includes pending spending, wealth stocks, capital demand in transit, and
  inventories-in-process (`1ᵀAx`). The `r < 1` family is a demand *stall* (surplus
  parks in the tracked hoard), deliberately repairing `io_economy`'s literal leak.
- Arrivals are exogenous endowments on a schedule (adoption stays `task_economy`'s
  job); owner behavior is fixed rules throughout — nothing optimises against the
  defenses (defended runs are upper bounds).
- **Absent:** price adjustment/substitution (the bracket), credit, a state actor,
  labor reallocation between sectors, adaptive owners.

**3. Classical result reproduced.** The households-only economy sits exactly at
its Leontief fixed point with money conservation exact (drift 0.0) and the
conserved-loop wealth/income ratio `σ_s/σ_d` (WP1 Prop. 2); the Gibrat-style
rich-get-richer mechanism of `compute_economy`'s ladder reappears as pro-rata
compounding above threshold.

**4. The dial.** `efficiency` (e) evolves by the capability-growth law
`e ← min(e_ceiling, e·(1 + growth_rate + rsi_strength·e))` — static baseline
(0, 0), first-order general improvement (g anchored to METR/Epoch doubling-time
*ranges* under a tick≈month reading, never point defaults), second-order RSI
(γ, arbitrary-but-swept). Initial `efficiency` (e) against the **pre-registered survival threshold
`e* = (δ/s + m)/v`** (WP1 Prop. 1, committed before any sweep ran; exported as
`survival_threshold()` so tests and experiments evaluate the paper's expression):
below the knee automation capital dies whatever its starting stock; above it,
it compounds until saturation. `recycle` (r) selects the output regime without
rescuing the ownership share (WP1 Prop. Decouple); `maintenance` sets the knee's
*location*, which is tuned — only existence and policy-dependence are claimed.

**5. Instrument.** `ai_wealth_share` (public capital counts on the human side —
that is the ownership mechanism's point), `human_income_share`, `output_late` /
`output_peak` (the decoupling pair), `money_drift` (the conservation probe,
`lab/analysis/conservation.py`). Benchmark `ScenarioSpec` wiring is future work;
the defense pair is `mechanisms.fiscal.make_ai_revenue_tax` (verbatim reuse —
taxable base = positive profit in `capital_income`) and the `ownership` (ω)
config dial (title diversion; output-neutral by construction, WP1 Prop. 4).

**5b. Resources.** Every named quantity in the state is typed in
`resources.py` (kind / substance / carrier / role in the money invariant) —
the environment's side of the type contract; a ladder rung keeps it exactly
in sync with the state.

**6. Lineage.** Forked from `io_economy` (2026-07-29, register R4) for the WP1
paper (Obsidian vault): replaced the exogenous recipe-editing schedule and the
stockless AI margin with owned capital stocks bearing upkeep; repaired the
reinvestment demand leak into a conservation-preserving stall. Spec was
referee-gated before implementation (13 defects raised and resolved).

**7. Status.** Live v0 (2026-07-29): validation ladder passing; conservation
exact to float32. Not yet a benchmark `ScenarioSpec`; sector-heterogeneous
capital and adaptive owners deferred by design.

**8. Instrument defect, found and fixed 2026-08-01.** `money_series` summed the
raw `last_reward` field. `ai_revenue_tax` writes a negative `-tax` record on
OWNER slots — a bookkeeping entry that no dynamic rule reads, since both `spend`
and the household wealth update consume `max(last_reward, 0)`. Counting it
double-subtracted the tax and reported a spurious ~2e-2 drift whenever a tax
mechanism sat in the slot, against ~4e-7 (float32 rounding) without one. The
dynamics were never at fault; the instrument was. Clipping `pending` at zero
restores exact conservation under the tax.

The defect survived because `test_conservation_at_every_closure` parametrizes
over the `r`-closure family only and never put a mechanism in the slot — the
probe's coverage gap, not its sensitivity. `test_conservation_holds_with_a_
mechanism_in_the_slot` now closes it. Found by porting this model to the web
playground (`eq-network/apps/playground`), where the same drift reproduced: a
reimplementation disagreeing with its source is worth more as a check than the
source's own green suite, which is the argument for keeping the port honest
rather than merely present.
