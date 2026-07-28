"""
Value Contagion — culture as something you catch (cultural register entry C2,
docs/cultural-register-design.md).

Humans and AI agents sit on one friendship network; each holds a cultural
variant tracked by origin. Two dials, both plain parameters: ``ai_homophily``
(axis S — does AI mix or self-segregate?) and ``p_advantage`` (axis P — how much
catchier are AI-origin ideas?). The four (S, P) corners produce four regimes —
pluralism, assimilation, parallel cultures, displacement — and only the last is
gradual cultural disempowerment. See ``examples/06_cultural_contagion.py``.

Registered as ``"value_contagion"`` in ``cilib.environments.REGISTRY``. Two forms:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (observe/step
  boundary) for plugging in your own broadcast policy.
- ``build_value_contagion(mechanisms, **cfg) -> EnvSpec`` — the closed
  convenience: the game closed with the catalog's constant-effort ``broadcast``
  policy.

    from cilib.environments import make_env
    env = make_env("value_contagion", ai_homophily=0.9, p_advantage=6.0)
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=16, n_steps=200)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.broadcast import BroadcastPolicy   # policies are inputs to games —
                                                     # agents sits below environments
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import ValueContagionConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics, make_reducers


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game. ``mechanisms`` are already-built Transforms (resolve names
    via ``cilib.mechanisms.REGISTRY`` yourself). Validates mechanism reads
    against the state schema at build time — a contract mismatch fails here,
    not mid-scan."""
    config = ValueContagionConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("value_contagion composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="value_contagion",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_value_contagion(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with constant full effort."""
    return close(build_game(mechanisms, **cfg), BroadcastPolicy(effort=1.0))


__all__ = [
    "ValueContagionConfig",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "make_reducers", "build_game",
    "build_value_contagion",
]
