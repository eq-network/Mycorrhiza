"""
Task Economy — jobs are task bundles; machines learn tasks one at a time; firms adopt
automation when it pays.

Register entry R3, the flagship rebuild (docs/model-register-design.md §3–4): aggregate
substitutability is *emergent* (task-level CES + an advancing frontier) and adoption is
*endogenous* (myopic cost comparison, ratcheted), which is what makes defenses change
behavior rather than bookkeeping. Skeleton status: production core + adoption + the
two limit anchors (Baumol bottleneck, full-automation collapse); demand side and
benchmark wiring are the remaining R3 work. Assumptions card: ``ASSUMPTIONS.md``.

Closed by ``agents.labor_supply.LaborSupplyPolicy`` — the same boundary as
``compute_economy``, so the two substrates are directly comparable.
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.labor_supply import LaborSupplyPolicy   # policies are inputs to games
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import TaskEconomyConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates mechanism reads against the state schema at build time."""
    config = TaskEconomyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("task_economy composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="task_economy",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_task_economy(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with the ``labor_supply`` policy."""
    game = build_game(mechanisms, **cfg)
    policy = LaborSupplyPolicy(wage_elasticity=game.config.wage_elasticity,
                               wage_ref=game.config.wage_ref,
                               noise=game.config.labor_noise)
    return close(game, policy)


__all__ = [
    "TaskEconomyConfig",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "build_game", "build_task_economy",
]
