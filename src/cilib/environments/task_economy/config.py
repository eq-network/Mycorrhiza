"""
Config for the Task Economy substrate — the register's flagship rebuild
(docs/model-register-design.md §3–4): every job is a bundle of tasks, machines learn
tasks one at a time, and firms adopt automation only when it pays.

The two moves that answer the compute_economy critique (register doc §2):

- **Substitutability is emergent, not assumed.** Output is a CES aggregate over K
  tasks with task-level complementarity ``sigma_task`` (< 1: tasks are complements —
  the world where humans are hardest to replace). What changes over time is the
  automation frontier, a *process*: capability ``beta_cap`` ramps on a schedule, and
  the *economics* decides how much of it is used.
- **Adoption is endogenous.** A capable task is automated iff compute is actually
  cheaper per unit output (``price_compute/a_M < wage/a_L``, judged myopically on last
  round's wage — a behavioral rule, not an optimization), and adoption ratchets
  (automated tasks stay automated). Compute is rented at an exogenous, declining
  price (elastic hardware supply, GATE-style); rental payments are the AI vendor's
  revenue. So a tax or a wage change moves *behavior* — adoption can pause.

Validation anchors (the two limits, tests/test_validation_ladder.py):

    Baumol bottleneck   sigma_task < 1, frontier stops short of 1: the un-automated
                        tasks are the bottleneck, wages RISE as automation frees labor
                        to concentrate on them (w ∝ (1−m)^((rho−1)/rho)·a_L), and the
                        labor share recovers as compute gets cheap.
    Full automation     the frontier crosses 1: the last tasks fall, labor exits
                        production entirely, wage collapses — rise-then-crash, which
                        is the shape the aggregate-CES model cannot produce.

Machines have **comparative advantage**: per-task machine productivity declines over
the task index, ``a_M(k) = a_M·(1 − hardness·k)`` — easy tasks are machine-friendly,
hard tasks barely. This is what makes adoption *smooth* (the profitable margin moves
with the price/wage ratio) instead of bang-bang; without it, one global profitability
flip automates most of the frontier in a single tick and crashes output through the
complements channel before wages can rise (observed in calibration, 2026-07-24).

Closed forms (mask over K tasks, measure-1/K each; labor L, rented compute C,
rho=(σ−1)/σ; optimal compute allocation across automated tasks absorbed into G):

    G    = (1/K)·Σ_automated a_M(k)^(rho/(1−rho))
    Y    = [ G^(1−rho)·C^rho + (1−m)^(1−rho)·(a_L·L)^rho ]^(1/rho)
    w    = Y^(1−rho)·(1−m)^(1−rho)·a_L^rho·L^(rho−1)          (0 once m = 1)
    C    = Y_prev·(G^(1−rho)/p)^(1/(1−rho))                    (myopic rental FOC)
    adopt task k  iff  capable(k)  and  p/a_M(k) < w/a_L       (ratcheted)
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TaskEconomyConfig:
    n_households: int = 20
    n_tasks: int = 50            # K discrete tasks standing in for the continuum

    # task-level technology
    sigma_task: float = 0.5      # elasticity across tasks; < 1 = complements
    a_L: float = 1.0             # labor productivity on manual tasks
    a_M: float = 1.0             # machine productivity on the EASIEST task
    hardness: float = 0.9        # a_M(k) = a_M·(1 − hardness·k_frac): comparative advantage

    # the two exogenous drivers (both schedules; everything else responds)
    cap_onset: int = 20          # capability frontier starts advancing here
    cap_rate: float = 0.01       # frontier advance per tick (fraction of tasks)
    cap_max: float = 1.0         # where the frontier stops (< 1 = Baumol regime)
    price_compute0: float = 2.0  # initial rental price of compute
    price_decline: float = 0.01  # per-tick relative decline after cap_onset

    # households (reuses agents.labor_supply.LaborSupplyPolicy)
    work_pref_center: float = 1.0
    work_pref_spread: float = 0.2
    work_pref_floor: float = 0.1
    wage_elasticity: float = 0.3
    wage_ref: float = 1.0
    labor_noise: float = 0.05

    eps: float = 1e-6
