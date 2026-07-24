# Assumptions card — `task_economy`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it.*

**1. What this says an economy is.** Jobs are bundles of tasks; machines learn tasks
one at a time as capability grows; firms automate a task only when it pays.

**2. Assumptions.**
- Output is a CES aggregate over K=50 tasks with task-level elasticity
  `sigma_task=0.5` — tasks are **complements**, the world where humans are hardest to
  replace. Aggregate substitutability is *emergent* from the frontier, never assumed
  (the direct answer to `compute_economy`'s critique).
- Machines have **comparative advantage**: per-task productivity
  `a_M(k) = a_M·(1 − hardness·k)` declines over the task index, so the profitable
  margin moves smoothly with the price/wage ratio (without it, adoption is bang-bang
  and crashes output through the complements channel — calibration note in
  `config.py`).
- **Adoption is endogenous and ratcheted**: task automated iff capable
  (`beta_cap` frontier, exogenous ramp) AND cheaper (`p/a_M(k) < w/a_L`, myopic on
  this round's wage). Compute is *rented* at an exogenously declining price (elastic
  hardware supply); the rental bill is the AI vendor node's revenue.
- Households supply labor (same `labor_supply` closure as `compute_economy` — the two
  substrates share a boundary and are directly comparable); myopic rental FOC on last
  round's output.
- **Absent (the remaining R3 work):** a demand side (whose preferences does output
  serve), compute-capital accumulation (rented, not owned), benchmark
  `ScenarioSpec` wiring, and any governance channel.

**3. Classical result reproduced** (`tests/test_validation_ladder.py`, measured
T=400, seed 0):
- *Manual limit*: no frontier ⇒ Y = a_L·L, wage = a_L, labor share = 1, exactly.
- *Baumol bottleneck* (frontier stops at 0.6): un-automated tasks are the bottleneck;
  automation makes remaining human work MORE valuable — wage 1.0 → **4.50**, labor
  share 0.85 late. **This substrate produces the opposite of disempowerment**, which
  is what makes finding disempowerment in it meaningful (register doc §10).
- *Full automation* (frontier crosses 1): wage rises to **3.05** while human tasks
  remain essential, then collapses to 0 the tick the last tasks fall — the
  rise-then-crash shape aggregate CES cannot produce.
- *Adoption pause*: compute priced high with no decline ⇒ capability reaches 1.0
  while adoption stays 0.0 and the economy is untouched — capability alone automates
  nothing.

**4. The dial.** `cap_max` selects the regime (< 1: Baumol/empowerment; = 1:
rise-then-collapse); `price_decline` sets the pace; `hardness` shapes how long the
comparative-advantage tail resists; `sigma_task` is the complementarity the whole
story runs through — swept, not assumed at a disempowerment-producing value.

**5. Instrument.** Not yet wired into the benchmark. Planned channel: work
preferences (shared boundary with `compute_economy` means `labor_dependence` runs
unchanged); the new readout this substrate adds is `adoption_gap_final`
(capability − adoption: how much frontier the economics declined).

**6. Lineage.** Originates here (2026-07-24, register R3 skeleton). Design sources:
Acemoglu–Restrepo task framework (comparative advantage, displacement at the margin),
Aghion–Jones–Jones (Baumol bottleneck growth), Epoch GATE (frontier-advances-with-
compute, endogenous adoption — their rational-expectations solution deliberately
replaced with myopic behavioral rules). Register siblings: `compute_economy`
(demoted), `io_economy` (σ=0 bracket).

**7. Status.** Skeleton v0 (2026-07-24): production core + endogenous adoption live,
4-rung ladder passing. Flagship-incomplete by design — demand side, capital loop, and
benchmark wiring are the open R3 work.
