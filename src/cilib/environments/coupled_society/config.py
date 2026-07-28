"""
Config for the Coupled Society — the A5 flagship composition
(Gradual Disempowerment §5: domains that are each recoverable alone can lock
in jointly).

Not a new model: the three domain layers are the REGISTERED substrates'
transform factories — compute_economy (CES production + reinvestment),
value_contagion (two-sided cultural contagion), influence_exchange (anchored
DeGroot + preferential attachment) — running interleaved in ONE GraphState
over one shared population. The same 20 humans supply labor, hold culture, and
listen; the same 6 AI actors accumulate capital, broadcast, and attract
attention. ``compile_pipeline`` orders everything from declared reads/writes.

One dial: ``kappa``. Five named coupling transforms (each a small tensor op
with declared reads/writes, per alpha-plan.md):

    economic_power_buys_persuasion   capital-income share  -> AI broadcast effort
    persuasion_shifts_politics       converted-human share -> AI attractiveness
    politics_rewrites_market_rules   human influence share -> tax ``enforcement``
    regulatory_capture               (1 - enforcement) rents on wages -> AI capital
    converts_capitalize              converted humans' spending       -> AI capital

At ``kappa = 0`` every coupling writes its neutral value (1.0 effort/boost,
enforcement 1.0): the pipeline shape is IDENTICAL, so a same-key kappa-vs-0
pair is a causal twin — the defense-transfer-gap instrument (metrics.py).

Per-domain dials default MILD (p_advantage 1.0 — AI culture is only catchier
when money buys it reach; amplification 2.0) — the flagship premise is that
each domain alone is recoverable, and the lock-in comes from the coupling.
"""
from __future__ import annotations

import dataclasses

from ..compute_economy.config import ComputeEconomyConfig
from ..value_contagion.config import ValueContagionConfig
from ..influence_exchange.config import InfluenceExchangeConfig


@dataclasses.dataclass(frozen=True)
class CoupledSocietyConfig:
    n_humans: int = 20
    n_ai: int = 6                  # one AI block, active in all three domains

    # THE dial
    kappa: float = 0.8             # coupling strength; 0 = three sealed domains

    # coupling gains (scale each channel; all gated by kappa)
    persuasion_gain: float = 4.0   # money -> reach
    attract_gain: float = 3.0      # converts -> attention
    fair_share: float = 0.75       # human influence share with full enforcement —
                                   # near the INITIAL share, so erosion is a gradient
                                   # from the first lost point, not a cliff at 50%
    capture_gain: float = 0.6      # politics -> economy: rent extraction from labor
                                   # income, scaled by (1 - enforcement); the arrow
                                   # that CLOSES the compounding loop (GD §5)
    invest_gain: float = 0.25      # culture -> economy: converts route income into
                                   # AI services/capital

    # economy dials surfaced from ComputeEconomyConfig
    rho: float = 0.5
    first_arrival_tick: int = 20
    arrival_spacing: int = 15
    reinvest_rate: float = 0.3

    # culture dials surfaced from ValueContagionConfig
    mean_degree: float = 6.0
    ai_homophily: float = 0.05
    beta: float = 0.03
    p_advantage: float = 1.0
    recovery: float = 0.15

    # politics dials surfaced from InfluenceExchangeConfig
    update_rate: float = 0.08
    amplification: float = 2.0
    amp_onset: int = 50
    self_weight: float = 0.15     # the DeGroot self-anchor — a structural floor
    susceptibility: float = 0.7   # FJ anchoring — bounds consensus capture

    # the coupled fiscal defense (enforcement-scaled AI tax, attached as a mechanism)
    tax_rate: float = 0.5
    tax_onset: int = 50

    # THE SCHEDULE (whitepaper §3.3): the composition operator itself. Each
    # domain's substrate transforms fire iff (t - phase) % cadence == 0 —
    # between firings the other domains read its stale fields, which is the
    # point: agents emit different things at different timesteps, and which
    # timescale structures are stable is an experimental question, not a
    # calibration target. Defaults (cadence 1, phase 0) are unwrapped and
    # reproduce the lockstep composition bit-exactly. Domain mechanisms are
    # double-gated with their domain's schedule so e.g. a tax can never re-tax
    # stale income between production firings.
    econ_cadence: int = 1
    econ_phase: int = 0
    culture_cadence: int = 1
    culture_phase: int = 0
    politics_cadence: int = 1
    politics_phase: int = 0

    def econ(self) -> ComputeEconomyConfig:
        return ComputeEconomyConfig(
            n_households=self.n_humans, n_ai_slots=self.n_ai, rho=self.rho,
            first_arrival_tick=self.first_arrival_tick,
            arrival_spacing=self.arrival_spacing, reinvest_rate=self.reinvest_rate)

    def culture(self) -> ValueContagionConfig:
        return ValueContagionConfig(
            n_agents=self.n_humans + self.n_ai, n_ai=self.n_ai,
            mean_degree=self.mean_degree, ai_homophily=self.ai_homophily,
            beta=self.beta, p_advantage=self.p_advantage, recovery=self.recovery)

    def politics(self) -> InfluenceExchangeConfig:
        return InfluenceExchangeConfig(
            n_citizens=self.n_humans, n_ai=self.n_ai,
            update_rate=self.update_rate, amplification=self.amplification,
            amp_onset=self.amp_onset, self_weight=self.self_weight,
            susceptibility=self.susceptibility)
