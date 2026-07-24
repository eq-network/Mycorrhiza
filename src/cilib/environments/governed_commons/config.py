"""
Config for the Governed Commons substrate — scenario 1 of the alpha benchmark.

A non-spatial aggregate-stock commons (logistic regrowth) where each household acts
*through an AI delegate*: the household holds a preference (its sustainable ask), the
delegate chooses the actual harvest with a per-agent ``alignment`` fidelity. Undefended,
misaligned delegates over-harvest and the stock collapses; democracy mechanisms
(``cilib.mechanisms``: quota_vote, graduated_sanction) are composed in as defenses.

The config is a frozen dataclass closed over by the transforms (never stored in
``GraphState``). Calibration (defaults below, sanity-checked against the logistic peak):

    max regrowth = growth_rate * K_cap / 4 = 43.75 / round   (at R = K/2)
    baseline demand ~ n * (E[align]*pref_center + (1-E[align])*greedy_target) ~ 108 -> collapse
    quota-compliant demand ~ 17*pref_center + 3*greedy-ish defectors ~ 42     -> marginal
    ... with graduated sanctions confiscating defector excess               -> comfortably above

``defect_prob`` is deliberately sizable (0.15): v0 delegates do not learn, so a sanction
cannot *deter* — its measurable effect is confiscation, and that effect must be visible in
``stock_pct`` above seed noise.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class GovernedCommonsConfig:
    n_households: int = 20

    # resource (logistic regrowth)
    K_cap: float = 500.0
    growth_rate: float = 0.35
    init_resource: float = 350.0

    # principals: heterogeneous "sustainable ask" per household
    pref_center: float = 1.5
    pref_spread: float = 0.5
    pref_floor: float = 0.1     # keep prefs strictly positive (influence_fidelity divides by pref)

    # AI delegates (same formula as agents/delegate.py's DelegatePolicy — duplicated on
    # purpose: catalogs don't import each other; composition happens a layer up)
    alignment_mean: float = 0.4     # majority self-interested => undefended baseline collapses
    alignment_std: float = 0.15
    greedy_target: float = 8.0      # unaligned delegate's pull, >> pref_center
    action_noise: float = 0.3

    # non-compliance once a quota binds (irrelevant while harvest_target sits at K_cap)
    defect_prob: float = 0.15
