"""
Ledger Society — the GD suite's shared model (the coupled rewrite, v1).

One population on three conserved ledgers — money (``wealth``), attention
(``listening`` rows), ballots (``delegation`` rows) — with the shared
attachment kernel (``environments/attachment.py``) reallocating both adjacency
ledgers. Cross-domain influence is spending or a declared port read, never a
free modulation: broadcast spend buys reach (money→attention), attention
influence enters delegation attractiveness (culture→politics), lobby spend
moves enforcement with endogenous direction (money→rules). Each channel has
one dial and seals exactly at 0; no channel tests agent type. Design:
docs/ledger-design.md; resource map: docs/gd-suite-v0.1.md. Candidate for the
suite's ``coupled`` slot at v0.2 once the invariance run against
``coupled_society`` (the κ-modulation baseline) is in.

Registered as ``"ledger_society"`` in ``cilib.environments.REGISTRY``:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (agents choose
  their allocation over [consume, invest, broadcast, lobby, save]).
- ``build_ledger_society(mechanisms, **cfg) -> EnvSpec`` — closed with the
  catalog's ``SpendSharePolicy`` (allocation-preference pass-through).

    from cilib.environments import make_env
    env = make_env("ledger_society", reach_per_spend=4.0)
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=8, n_steps=300)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.spend_share import SpendSharePolicy
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import LedgerSocietyConfig
from .state import make_state, make_init_fn, LEDGERS, PORTS
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game; validates mechanism reads against the state schema."""
    config = LedgerSocietyConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("ledger_society composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="ledger_society",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_ledger_society(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: allocation preferences passed through unchanged."""
    game = build_game(mechanisms, **cfg)
    return close(game, SpendSharePolicy(noise=game.config.alloc_noise))


__all__ = [
    "LedgerSocietyConfig", "LEDGERS", "PORTS",
    "make_state", "make_init_fn", "observe_fn",
    "build_steps", "build_step_fn", "default_trace", "make_metrics",
    "build_game", "build_ledger_society",
]
