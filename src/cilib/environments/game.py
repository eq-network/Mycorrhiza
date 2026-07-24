"""
The open-environment contract: ``GameSpec`` — a game form awaiting policies.

``EnvSpec`` (spec.py) is a *closed* system: its ``round_fn`` contains the decision rule.
A ``GameSpec`` is the same environment with the agent↔environment boundary made explicit
(the Gym/MDP sense, functional style a la Gymnax/JaxMARL):

    observe_fn : state -> obs                 per-agent observations, leading axis N
    step_fn    : (state, actions, key) -> state    everything that is NOT a decision —
                 internally still a compiled transform pipeline (mechanisms included)
    rewards    : a VIEW of state (node_attrs["last_reward"]) — payoff-rule mechanisms
                 (sanctions, taxation) transform state, so no separate reward plumbing

``close(game, policy)`` attaches a policy at the boundary and returns an ordinary
``EnvSpec`` — the closed world is a special case of the open one, so everything built
on ``EnvSpec`` (run/run_batch/evaluate, the benchmark harness) keeps working. Design
record: docs/game-boundary-design.md.

Layering note: policies are *inputs* to games, so the ``agents`` catalog sits below
``environments`` in the import DAG — an environment builder may import a policy from
``cilib.agents`` to provide its default closure.

All simultaneous-move (parallel API): actions and observations are arrays with a
leading agent axis, never dicts of dicts — this is what keeps ``vmap`` trivial.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.scan import TraceFn
from .spec import EnvSpec, MetricFn

# A policy is (obs_row, key) -> action (cilib.core.protocols.Policy); close() vmaps it
# over the agent axis, so per-agent parameters live inside obs or the policy object.
Policy = Callable[[Any, Any], Any]


@dataclasses.dataclass(frozen=True)
class GameSpec:
    """An open environment: dynamics + boundary, no decision rule."""

    name: str
    config: Any
    init_fn: Callable[[Any], GraphState]
    observe_fn: Callable[[GraphState], Any]
    step_fn: Callable[[GraphState, Any, Any], GraphState]
    trace_fn: Optional[TraceFn] = None
    metrics: Dict[str, MetricFn] = dataclasses.field(default_factory=dict)

    def rewards(self, state: GraphState):
        """Per-agent rewards as a view of state (the payoff-rule hook point)."""
        return state.node_attrs["last_reward"]


def close(game: GameSpec, policy: Policy) -> EnvSpec:
    """Attach a policy at the boundary: open game -> closed ``EnvSpec``.

    One round = observe -> act (policy, vmapped over agents, per-agent keys) -> step.
    """
    def round_fn(state: GraphState, t, key) -> GraphState:
        k_act, k_step = jr.split(key)
        n_agents = state.node_types.shape[0]
        obs = game.observe_fn(state)
        actions = jax.vmap(policy)(obs, jr.split(k_act, n_agents))
        return game.step_fn(state, actions, k_step)

    return EnvSpec(
        name=game.name,
        config=game.config,
        init_fn=game.init_fn,
        round_fn=round_fn,
        trace_fn=game.trace_fn,
        metrics=game.metrics,
    )


def validate_reads(transforms: Sequence[Any], state: GraphState) -> List[str]:
    """Registration-time check: every declared read must be an existing state field or
    an earlier transform's write. Returns human-readable issues (empty = OK).

    This is the composition-safety gap the pipeline compiler doesn't cover: the
    compiler orders hazards among *declared* fields, but nothing else checks that a
    mechanism's contract matches the environment's schema — without this, a mismatch
    surfaces as a KeyError mid-``lax.scan`` instead of a clear error at build time.
    """
    available = (set(state.node_attrs) | set(state.global_attrs)
                 | set(state.adj_matrices) | set(state.edge_attrs))
    issues = []
    for t in transforms:
        name = getattr(t, "name", getattr(t, "__name__", "?"))
        missing = frozenset(getattr(t, "reads", frozenset())) - available
        if missing:
            issues.append(f"{name}: reads {sorted(missing)} not in state schema "
                          f"or any earlier transform's writes")
        available |= set(getattr(t, "writes", frozenset()))
    return issues
