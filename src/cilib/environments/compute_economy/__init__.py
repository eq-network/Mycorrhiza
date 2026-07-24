"""
Compute Economy — the classical macro-ABM of economic disempowerment
(alpha benchmark scenario 2, Gradual Disempowerment §2).

Registered as ``"compute_economy"`` in ``cilib.environments.REGISTRY``. Two forms:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (observe/step boundary).
- ``build_compute_economy(mechanisms, **cfg) -> EnvSpec`` — the closed convenience:
  the game closed with the catalog's ``labor_supply`` household policy.

    from cilib.environments import make_env
    env = make_env("compute_economy", rho=0.5)          # σ=2: substitutes — labor share decays
    env = make_env("compute_economy", rho=0.0)          # Cobb-Douglas: constant share (validation)
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=32, n_steps=300)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.labor_supply import LaborSupplyPolicy   # policies are inputs to games
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import ComputeEconomyConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import (
    make_metrics, shift_work_pref, make_work_pref_shift, late_log_output,
    make_window_log_output, make_window_log_labor, mean_labor_share, per_agent_income,
)


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates mechanism reads against the state schema at build time."""
    config = ComputeEconomyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("compute_economy composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="compute_economy",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_compute_economy(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with the ``labor_supply`` policy."""
    game = build_game(mechanisms, **cfg)
    policy = LaborSupplyPolicy(wage_elasticity=game.config.wage_elasticity,
                               wage_ref=game.config.wage_ref,
                               noise=game.config.labor_noise)
    return close(game, policy)


__all__ = [
    "ComputeEconomyConfig",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "build_game", "build_compute_economy",
    "shift_work_pref", "make_work_pref_shift", "late_log_output",
    "make_window_log_output", "make_window_log_labor",
    "mean_labor_share", "per_agent_income",
]
