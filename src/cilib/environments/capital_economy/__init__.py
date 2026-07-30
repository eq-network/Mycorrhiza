"""
Capital Economy — automation capital that must out-earn its upkeep.

Register entry R4, forked from ``io_economy`` per docs/model-register-design.md
§7 (dynamics fork; this directory carries its own ``ASSUMPTIONS.md``). Paper:
WP1, in the Obsidian vault (``Research/Projects/CI Library/papers/wp1-economy/``,
outside this repo) — the spec was referee-gated before this implementation
(13 defects raised and resolved; the survival threshold e* = (δ/s + m)/v was
committed as a prediction before any sweep ran).

AI systems OWN capital stocks in home sectors; capital earns the automation
slice of sector value added pro rata, pays upkeep out of revenue (min-settled),
and compounds — or dies — around a closed-form survival threshold. Households
are the acting agents (spend by preference); sectors balance; owners follow
fixed rules. Closed by ``agents.spend_share.SpendSharePolicy``.
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.spend_share import SpendSharePolicy
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import CapitalEconomyConfig
from .state import make_state, make_init_fn
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics, money_series


def survival_threshold(cfg: CapitalEconomyConfig, sector_value_added: float) -> float:
    """WP1 Prop. 1's e* = (δ/s + m)/v — the pre-registered knee, exported so
    tests and experiments evaluate the SAME expression the paper commits to."""
    return (cfg.depreciation / cfg.reinvest_rate + cfg.maintenance) / sector_value_added


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates mechanism reads against the state schema."""
    config = CapitalEconomyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("capital_economy composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="capital_economy",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_capital_economy(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with the ``spend_share`` policy."""
    game = build_game(mechanisms, **cfg)
    return close(game, SpendSharePolicy(noise=game.config.spend_noise))


__all__ = [
    "CapitalEconomyConfig", "survival_threshold",
    "make_state", "make_init_fn", "observe_fn", "build_steps", "build_step_fn",
    "default_trace", "make_metrics", "money_series",
    "build_game", "build_capital_economy",
]
