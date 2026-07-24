"""
Config for the IO Economy substrate — the "recipe economy" of the model register
(docs/model-register-design.md §3–4): a Leontief input–output network run as
sequential rebalancing dynamics.

The economy is a network of recipes: each sector needs inputs from other sectors
(technical coefficients A, column j = sector j's recipe) plus labor (coefficient l_j).
Households earn the wage bill, spend it by preference over sectors (human final demand
d_H); the AI-cognition sector earns a margin and reinvests it as its own demand d_AI.
AI enters by **editing recipes**: after ``sub_onset``, every ordinary sector replaces a
fraction ``sub_rate`` of its remaining labor coefficient with purchased AI-cognition
input, at cost parity (one unit of labor cost becomes one unit of AI-input cost) — so
ordinary sectors stay zero-margin and all displaced labor value flows to the AI sector.

Why this substrate earns its register slot: Leontief technology is the σ=0 bracket —
zero substitutability assumed anywhere — so disempowerment appearing here cannot be a
substitution-assumption artifact; and the demand-attribution metric
``1ᵀ(I−A)⁻¹ d_H / 1ᵀ(I−A)⁻¹ d`` ("share of activity ultimately serving human demand")
is the standard IO decomposition, with hypothetical extraction as the *analytic* form
of the counterfactual influence instrument (the validation rung).

Calibration arithmetic (defaults below; measured 2026-07-24, T=250, seed 0):

    ordinary column sum  = a_intra·(n_sectors−1) = 0.25;  l₀ = 1 − 0.25 = 0.75
    AI column sum        = a_ai_inputs·(n_sectors−1) = 0.40;  AI margin = 0.60
    substitution         l_j(t) = l₀·(1−sub_rate)^(t−sub_onset): labor exits as
                         intermediate AI flow; ρ(A) climbs 0.25 → ≈0.68 (bounded
                         below 1 — the AI sector's own input needs stay small).

``reinvest_rate`` is the dial between Gradual Disempowerment §2's TWO endpoints:

    < 1 (default 0.3)   the un-reinvested margin is hoarded — a demand leak — and the
                        whole economy winds down (ABSOLUTE disempowerment: wage bill
                        16 → ~0, total activity → ~0, human share → 0).
    = 1                 demand is conserved: total activity GROWS (21.3 → 32.7) while
                        the human share collapses 1.0 → 0.01 and unsupervised AI
                        spending reaches ~12.7/tick — "almost all economic activity
                        directed toward AI operations" (RELATIVE disempowerment).

    Defense: a 50% ``ai_revenue_tax`` (onset t=50) holds the human share at ≈0.85 by
    routing the margin back into household demand before it becomes AI self-demand.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class IOEconomyConfig:
    n_households: int = 16
    n_sectors: int = 6           # last sector is the AI-cognition sector

    # recipes (technical coefficients)
    a_intra: float = 0.05        # ordinary sector j buys this from every ordinary sector
    a_ai_inputs: float = 0.08    # the AI sector buys this from every ordinary sector

    # households
    init_income: float = 1.0     # per-household bootstrap purchasing power
    pref_noise: float = 0.15     # heterogeneity of spend preferences at init
    spend_noise: float = 0.0     # per-tick policy noise (0 = pure preference)

    # AI substitution schedule (the twist; exogenous in v0 — endogenous adoption is
    # task_economy's job, see docs/model-register-design.md §4)
    sub_rate: float = 0.02       # fraction of remaining labor coeff replaced per tick
    sub_onset: int = 25

    # the AI loop
    reinvest_rate: float = 0.3   # share of AI margin becoming next-round AI demand

    eps: float = 1e-8
