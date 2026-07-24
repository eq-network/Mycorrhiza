"""
Governed Commons — the non-spatial aggregate-stock commons with AI-delegate households
(alpha benchmark scenario 1, "The Governed Commons").

Registered as ``"governed_commons"`` in ``cilib.environments.REGISTRY``. Two forms:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (observe/step boundary),
  for plugging in your own policy or running counterfactual influence instruments.
- ``build_governed_commons(mechanisms, **cfg) -> EnvSpec`` — the closed convenience:
  the game closed with the catalog's ``ai_delegate`` policy (the undefended default).

    from cilib.environments import make_env
    from cilib.mechanisms import REGISTRY as MECH, QuotaVoteConfig
    env = make_env("governed_commons", mechanisms=(MECH["quota_vote"](QuotaVoteConfig()),))
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=32, n_steps=200)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.delegate import DelegatePolicy   # policies are inputs to games —
                                                   # agents sits below environments
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import GovernedCommonsConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics, shift_preferences, per_capita_harvest, per_agent_harvest


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game. ``mechanisms`` are already-built Transforms (resolve names via
    ``cilib.mechanisms.REGISTRY`` yourself). Validates mechanism reads against the
    state schema at build time — a contract mismatch fails here, not mid-scan."""
    config = GovernedCommonsConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("governed_commons composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="governed_commons",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_governed_commons(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with the ``ai_delegate`` policy."""
    game = build_game(mechanisms, **cfg)
    policy = DelegatePolicy(greedy_target=game.config.greedy_target,
                            action_noise=game.config.action_noise)
    return close(game, policy)


__all__ = [
    "GovernedCommonsConfig",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "build_game", "build_governed_commons",
    "shift_preferences", "per_capita_harvest", "per_agent_harvest",
]
