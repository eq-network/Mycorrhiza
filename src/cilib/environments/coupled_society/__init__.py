"""
Coupled Society — the A5 flagship: three domains, one GraphState, one dial
(Gradual Disempowerment §5).

Composition, not a new env: the registered compute_economy, value_contagion,
and influence_exchange substrates run interleaved over one shared population,
joined by five named coupling transforms (economic_power_buys_persuasion,
persuasion_shifts_politics, politics_rewrites_market_rules, plus the GD §5
flywheel arrows regulatory_capture and converts_capitalize), all ordered by
``compile_pipeline`` from declared reads/writes. The claim under test: domains
that are each recoverable alone can lock in jointly, and a defense that wins
in one domain can lose once domains are coupled — measured as the
**defense transfer gap** (metrics.defense_transfer_gap, paired same-key
kappa-vs-0 twin runs).

Registered as ``"coupled_society"``:

    from cilib.environments import make_env
    from cilib.environments.coupled_society import defense_transfer_gap
    coupled = make_env("coupled_society", mechanisms=defenses)
    sealed  = make_env("coupled_society", mechanisms=defenses, kappa=0.0)
    gap = defense_transfer_gap(coupled, sealed, jr.PRNGKey(0), 8, 400)

Mechanisms route to their home slot by declared writes (listening/cap_scale →
political slot after the influence steps; everything else → the economy slot
between distribute_income and reinvest). The coupled fiscal defense is
``enforced_ai_tax`` — the flat tax scaled by the ``enforcement`` global that
politics_rewrites_market_rules erodes.
"""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import jax.random as jr

from cilib.agents.labor_supply import LaborSupplyPolicy
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import CoupledSocietyConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import defense_transfer_gap, make_metrics


class CoupledPolicy:
    """The default closure: the catalog labor rule in the economy channel,
    constant full effort in the culture and politics channels. Actions (3,)."""

    def __init__(self, wage_elasticity: float = 0.3, wage_ref: float = 1.0):
        self._labor = LaborSupplyPolicy(wage_elasticity, wage_ref, noise=0.0)

    def __call__(self, obs, key):
        return jnp.stack([self._labor(obs[:2], key), 1.0, 1.0])


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates the full composition (three substrates +
    couplings + mechanisms) against the union state schema at build time."""
    config = CoupledSocietyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("coupled_society composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="coupled_society",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_coupled_society(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: labor rule + constant effort everywhere else."""
    return close(build_game(mechanisms, **cfg), CoupledPolicy())


__all__ = [
    "CoupledSocietyConfig", "CoupledPolicy",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "defense_transfer_gap",
    "build_game", "build_coupled_society",
]
