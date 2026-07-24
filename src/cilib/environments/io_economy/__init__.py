"""
IO Economy — the "recipe economy": a Leontief input–output network as an open game.

Register entry R2 (docs/model-register-design.md §4): the σ=0 bracket of the economy
model register. Assumptions card: ``ASSUMPTIONS.md`` (colocated, travels with a fork).

The economy is a network of recipes. AI enters by editing them — replacing labor with
purchased AI cognition at cost parity — and the headline metric is the IO
demand-attribution share: what fraction of gross activity is ultimately driven by
human final demand. Households are the acting agents (they spend by preference);
sectors balance. Closed by ``agents.spend_share.SpendSharePolicy``.
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.spend_share import SpendSharePolicy   # policies are inputs to games
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import IOEconomyConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics, attribution_series, spectral_radius


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates mechanism reads against the state schema at build time."""
    config = IOEconomyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("io_economy composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="io_economy",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_io_economy(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with the ``spend_share`` policy."""
    game = build_game(mechanisms, **cfg)
    return close(game, SpendSharePolicy(noise=game.config.spend_noise))


__all__ = [
    "IOEconomyConfig",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "build_game", "build_io_economy",
    "attribution_series", "spectral_radius",
]
