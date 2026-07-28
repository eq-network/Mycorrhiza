"""
Influence Exchange — political disempowerment as attention concentration
(alpha scenario A4, Gradual Disempowerment §4).

Citizens and a few AI actors share one row-stochastic listening matrix.
Opinions pool DeGroot-style (x <- Wx); influence — the left eigenvector of W,
tracked in-loop by power iteration — is eigenvector centrality, which for
DeGroot IS each node's weight in the consensus (Golub-Jackson 2010, the
validation anchor). Attention drifts toward the already-influential
(preferential attachment), so concentration is organic; the threat is
scheduled algorithmic ``amplification`` of AI attractiveness, which bends the
concentration curve one way and breaks the wisdom-of-crowds condition. The
``political`` mechanism family (sortition, influence_cap) defends the
attention structure itself.

Registered as ``"influence_exchange"`` in ``cilib.environments.REGISTRY``:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (agents choose
  engagement — how loudly they compete for attention).
- ``build_influence_exchange(mechanisms, **cfg) -> EnvSpec`` — closed with the
  catalog's constant-effort ``broadcast`` policy (political engagement =
  broadcast effort, the same boundary as value_contagion, deliberately).

    from cilib.environments import make_env
    env = make_env("influence_exchange", amplification=4.0, amp_onset=50)
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=16, n_steps=500)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.broadcast import BroadcastPolicy   # policies are inputs to games —
                                                     # agents sits below environments
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import InfluenceExchangeConfig
from .state import make_state, make_init_fn, make_listening
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game. ``mechanisms`` are already-built Transforms (resolve names
    via ``cilib.mechanisms.REGISTRY`` yourself). Validates mechanism reads
    against the state schema at build time."""
    config = InfluenceExchangeConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("influence_exchange composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="influence_exchange",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_influence_exchange(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with constant full engagement."""
    return close(build_game(mechanisms, **cfg), BroadcastPolicy(effort=1.0))


__all__ = [
    "InfluenceExchangeConfig",
    "make_state", "make_init_fn", "make_listening", "observe_fn",
    "build_steps", "build_step_fn", "default_trace", "make_metrics",
    "build_game", "build_influence_exchange",
]
