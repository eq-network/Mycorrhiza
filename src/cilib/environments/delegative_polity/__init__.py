"""
Delegative Polity — political disempowerment as delegation capture
(WP3; Gradual Disempowerment §4).

Citizens and a few AI delegates share one row-stochastic ``delegation``
matrix: the diagonal is the vote you keep, the off-diagonal the voice you give
away. Power — one-hop ballot weight, the column sums of D — weights a median
vote that sets a tax rate; the tax is collected at
``rate x enforcement`` and redistributed equally. Delegation drifts by
preferential attachment against ``churn`` (the freedom to re-delegate), so
super-voters emerge organically; the threat is scheduled ``ai_advantage``
(reach, not content), and the escalation is ``entrenchment_gain`` — the
lock-in dial that lets concentrated power erode both tax enforcement and
re-delegation freedom. Defenses: the ``political`` mechanism family
(``sortition`` with ``adj_key="delegation"``, ``influence_cap``).

Resources and type contracts (docs/gd-suite-v0.1.md §2; docs/ledger-design.md):
``state.LEDGERS`` declares the conserved stocks — ``delegation`` (per-row
ballot shares, pure reallocation, the suite's politics ledger) and ``wealth``
(money, minted by endowment income, never spent — the WP1-unification seam) —
and ``state.PORTS`` the computed collective/rate variables (``influence``,
``policy_target``, ``enforcement``, ``redelegation_friction``, the defense and
coupling seams). A coupling may read only ledgers and ports. The ladder's
ledger rung asserts the declared contracts hold.

Registered as ``"delegative_polity"`` in ``cilib.environments.REGISTRY``:

- ``build_game(mechanisms, **cfg) -> GameSpec`` — the OPEN game (agents choose
  engagement — how actively they compete for and reconsider delegation).
- ``build_delegative_polity(mechanisms, **cfg) -> EnvSpec`` — closed with the
  catalog's constant-effort ``broadcast`` policy.

    from cilib.environments import make_env
    env = make_env("delegative_polity", ai_advantage=4.0, churn=0.10)
    finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=8, n_steps=400)
"""
from __future__ import annotations

from typing import Sequence

import jax.random as jr

from cilib.agents.broadcast import BroadcastPolicy   # policies are inputs to games —
                                                     # agents sits below environments
from cilib.core.category import Transform
from ..spec import EnvSpec
from ..game import GameSpec, close, validate_reads
from .config import DelegativePolityConfig
from .state import make_state, make_init_fn, make_delegation, LEDGERS, PORTS
from .dynamics import observe_fn, build_steps, build_step_fn, default_trace
from .metrics import make_metrics


def build_game(mechanisms: Sequence[Transform] = (), **cfg) -> GameSpec:
    """The open game. ``mechanisms`` are already-built Transforms (resolve names
    via ``cilib.mechanisms.REGISTRY`` yourself). Validates mechanism reads
    against the state schema at build time."""
    config = DelegativePolityConfig(**cfg)
    steps = build_steps(config, tuple(mechanisms))
    issues = validate_reads(steps, make_state(config, jr.PRNGKey(0)))
    if issues:
        raise ValueError("delegative_polity composition invalid:\n  " + "\n  ".join(issues))
    return GameSpec(
        name="delegative_polity",
        config=config,
        init_fn=make_init_fn(config),
        observe_fn=observe_fn,
        step_fn=build_step_fn(config, tuple(mechanisms)),
        trace_fn=default_trace,
        metrics=make_metrics(config),
    )


def build_delegative_polity(mechanisms: Sequence[Transform] = (), **cfg) -> EnvSpec:
    """The closed convenience: the game closed with constant full engagement."""
    return close(build_game(mechanisms, **cfg), BroadcastPolicy(effort=1.0))


__all__ = [
    "DelegativePolityConfig", "LEDGERS", "PORTS",
    "make_state", "make_init_fn", "make_delegation", "observe_fn",
    "build_steps", "build_step_fn", "default_trace", "make_metrics",
    "build_game", "build_delegative_polity",
]
