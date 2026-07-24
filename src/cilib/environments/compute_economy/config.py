"""
Config for the Compute Economy substrate — scenario 2 of the alpha benchmark
(economic disempowerment, Gradual Disempowerment §2).

A classical macro-ABM: households supply labor; a production sector combines labor L and
compute C via CES; wages are labor's marginal product; compute's earnings reinvest into
more compute. AI actors (pre-allocated node slots) arrive on a schedule and compound.
The **substitutability assumption is the disempowerment dial**: with elasticity of
substitution σ = 1/(1−ρ) > 1 (ρ ∈ (0,1)), growing compute drives the labor share of
income toward 0; the Cobb-Douglas limit ρ = 0 gives the textbook constant labor share α
(the validation anchor).

Calibration arithmetic (validated in a numpy prototype, 2026-07-14):

    Y    = A · (α·L^ρ + (1−α)·C^ρ)^(1/ρ)          CES; ρ=0 branch: Y = A·L^α·C^(1−α)
    wage = α·A^ρ·(Y/L)^(1−ρ)                       marginal product of labor
    r    = (1−α)·A^ρ·(Y/C)^(1−ρ)                   marginal product of compute
    wage·L + r·C = Y  exactly (Euler)              income accounting closes

    Defaults (ρ=0.5 ⇒ σ=2, 20 households, 6 AI slots arriving t=20..95):
    labor share 0.995 → ~0.15 over 300 ticks (the dependence-decay curve); income
    concentration rises ~0.05 → ~0.36 (rich-get-richer from reinvestment); the absolute
    wage RISES throughout — relative power collapses without absolute immiseration.
    ρ=0: labor share ≡ α = 0.6 at every tick. No AI arrivals: constant steady state.
    50% AI-revenue tax from t=50 REVERSES the labor-share decline (taxing capital income
    *before* reinvestment slows the compounding: growth factor 1 − depreciation +
    reinvest_rate·(1−tax)·r drops below 1 as r diminishes).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ComputeEconomyConfig:
    n_households: int = 20
    n_ai_slots: int = 6          # pre-allocated, dormant until scheduled arrival

    # production (CES); rho in [0, 1): 0 = Cobb-Douglas (validation), 0.5 = substitutes
    A: float = 1.0
    alpha: float = 0.6
    rho: float = 0.5
    eps: float = 1e-3            # division/power guard

    # AI arrivals: slot i activates at first_arrival_tick + i * arrival_spacing
    first_arrival_tick: int = 20
    arrival_spacing: int = 15
    initial_ai_capital: float = 5.0

    # compute dynamics
    reinvest_rate: float = 0.3
    depreciation: float = 0.05

    # households
    work_pref_center: float = 1.0
    work_pref_spread: float = 0.2
    work_pref_floor: float = 0.1
    wage_elasticity: float = 0.3
    wage_ref: float = 1.0
    labor_noise: float = 0.05
