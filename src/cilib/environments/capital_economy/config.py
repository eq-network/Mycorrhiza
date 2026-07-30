"""
Config for the Capital Economy substrate — the WP1 fork of ``io_economy``
(WP1 paper §3, referee-accepted 2026-07-29; the paper lives in the Obsidian vault).

What forked and why (ASSUMPTIONS.md field 6 has the card version): ``io_economy``
automates by an *exogenous schedule* editing recipes, and its AI actor is a sector
with a flow margin — no stock, nothing that has to pay to exist. This fork gives
automation an OWNER: AI systems hold capital ``K_i`` in a home sector; capital
contributes capacity ``e·K`` against one unit of human capacity (automation share
``a_j = eK_j/(eK_j+1)``), earns the capital slice ``a_j·v_j·x_j`` pro rata, and must
pay upkeep ``m·K`` out of revenue before profit. The two clips in the update

    K ← [(1−δ)K + (1−ω)·r·s·[π]₊]₊         (loss charged to the stock when π<0)

are the fork's substance: below the survival threshold  e* = (δ/s + m)/v  capital
decays to extinction whatever its starting stock (WP1 Prop. 1 — committed as a
prediction before any sweep ran); above it, it compounds until saturation.

Money is conserved at every closure: upkeep and investment are machine-sector
purchases (demand, not leakage), AI consumption spends the tracked hoard, and the
``r < 1`` closure family is a demand *stall* — unspent surplus parks in the hoard —
not ``io_economy``'s literal leak, which this fork deliberately repairs.

Parameter typing (WP1 Table 2 is authoritative): ``e``, ``s``, ``r``, ``tau``
(via mechanism), ``omega`` are arbitrary-but-swept; ``m``, ``delta``, ``a_*``,
``sigma_s``/``sigma_d``, ``c_a`` are tuned-for-legibility (the σ ratio pins
W_H/Y exactly and is flagged tuned-but-exactly-load-bearing).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CapitalEconomyConfig:
    n_households: int = 20
    n_sectors: int = 6           # sector 0 (node index H) is the machine sector
    n_owners: int = 6            # AI owner i's home sector is i % n_sectors

    # recipes: hub (machines feed every other sector) + chain (j-1 -> j, j >= 2)
    a_machines: float = 0.15
    a_chain: float = 0.12

    # households
    init_income: float = 1.0
    pref_noise: float = 0.15
    spend_noise: float = 0.0
    sigma_s: float = 0.08        # saving out of income
    sigma_d: float = 0.03        # wealth drawdown (closes the no-AI loop exactly)

    # capital physics (the knee lives in these three plus e)
    efficiency: float = 0.55     # e — capacity per unit K (the knee axis); INITIAL
                                 # value: e evolves by the growth law below
    maintenance: float = 1.5     # m — upkeep per unit K, paid out of revenue only
    reinvest_rate: float = 0.5   # s
    depreciation: float = 0.05   # delta

    # capability growth: e <- min(e_ceiling, e·(1 + g + γ·e)) once per tick.
    # (0,0) = static (the WP1 baseline); (g,0) = first-order general improvement
    # (constant doubling time — anchor to METR/Epoch RANGES, never defaults);
    # (g,γ>0) = second-order recursive self-improvement (doubling time shrinks
    # with capability). The end-state law h* ≈ e*/e then makes the growth type
    # the shape of the collapse: none / exponential glide / finite-time cliff.
    growth_rate: float = 0.0     # g — first-order improvement per tick
    rsi_strength: float = 0.0    # γ — second-order (RSI) term, scales with e
    e_ceiling: float = 64.0      # numerical cap on e (float32 hygiene; stated in paper)

    # arrivals (exogenous entry; adoption stays task_economy's job)
    first_arrival: int = 40
    arrival_spacing: int = 25
    init_capital: float = 0.5

    # closures and ownership
    recycle: float = 1.0         # r — scales AI discretionary outlays; rest stalls in the hoard
    consume_rate: float = 0.02   # c_A — AI consumption out of the hoard
    ownership: float = 0.0       # omega — slice of investment whose TITLE goes public
    # fund disposal design (WP1 Prop. 4's two measured regimes):
    #   False (dividend fund): pays ALL profit to households — diversion is a
    #     partial transfer; interior ownership split, aggregate capital lower.
    #   True (mirror fund): reinvests the same s-slice as private owners — the
    #     RATCHET: common per-unit profit + the multiplier gap (fund keeps its
    #     whole reinvestment AND takes omega of private's) drive the public
    #     share toward 1 and private capital to extinction at rate ~(1 - ω·δ).
    pub_mirror: bool = False

    eps: float = 1e-8
