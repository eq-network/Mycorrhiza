"""
Democracy family — collective-choice and enforcement mechanisms.

Ostrom's core loop (Governing the Commons, 1990): the group sets its own rules
(collective-choice arrangements) and enforces them with sanctions graduated in the size
of the violation. Two composable entries:

- ``quota_vote``: direct democracy — a quantile (default median) of the ``vote`` field
  becomes the global ``policy_target``. Timing (re-vote cadence, onset) belongs to the
  schedule, not the mechanism: wrap with ``core.schedule.scheduled`` (the benchmark's
  composition principle). Both field names are deliberately generic
  (IAD rule types: ``vote`` feeds an *aggregation rule*; ``policy_target`` is the rule it
  enacts — a harvest quota here, an emissions cap or tax level elsewhere). The
  *environment* decides who casts votes (governed_commons v0: the human principals).
- ``graduated_sanction``: penalizes over-quota extraction proportionally to the violation
  and claws a share of the excess back into the common pool. Confiscation (not just a
  reward penalty) is deliberate: against non-learning agents a pure reward penalty would
  be invisible in the resource trajectory.

Both follow the family contract (disjoint writes: {policy_target} vs {last_reward,
sanction, resource_level}), so ``compile_pipeline`` composes them in any order without
conflict. Composed after the substrate steps, a mechanism reads this round's outcome and
sets next round's rules — on a re-vote tick the sanction judges against the just-enacted
quota (a one-tick founding artifact at t=0; immaterial in v0, where votes are static).

Templates: ``lab/paradigms/polycentric/transforms.py`` (quota/sanction math),
EXTENDING.md's democracy example (field names). NOT ``market.py`` (pre-``@transform``).
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform


@dataclasses.dataclass(frozen=True)
class QuotaVoteConfig:
    quantile: float = 0.5   # 0.5 = median vote


def make_quota_vote(cfg: QuotaVoteConfig):
    """Quantile vote over ``node_attrs["vote"]`` → ``global_attrs["policy_target"]``.

    Pure rule, no timing: WHEN it fires belongs to the schedule —
    ``core.schedule.scheduled(make_quota_vote(cfg), cadence=5)`` re-votes every 5 ticks.
    """

    @transform(reads=["vote"], writes=["policy_target"])
    def quota_vote(state: GraphState) -> GraphState:
        voted = jnp.quantile(state.node_attrs["vote"], cfg.quantile)
        return state.update_global_attr("policy_target", voted)
    return quota_vote


@dataclasses.dataclass(frozen=True)
class SanctionConfig:
    sanction_strength: float = 1.5   # reward penalty per unit over-quota
    confiscate_rate: float = 0.5     # share of the excess returned to the common pool


def make_graduated_sanction(cfg: SanctionConfig):
    """Graduated sanction (Ostrom): per-agent penalty ∝ the individual's violation, plus
    confiscation of a share of the total excess back into ``resource_level``."""

    @transform(reads=["last_harvest", "policy_target", "last_reward", "resource_level"],
               writes=["last_reward", "sanction", "resource_level"])
    def graduated_sanction(state: GraphState) -> GraphState:
        over = jnp.maximum(
            state.node_attrs["last_harvest"] - state.global_attrs["policy_target"], 0.0)
        penalty = cfg.sanction_strength * over
        state = state.update_node_attrs("sanction", penalty)
        state = state.update_node_attrs(
            "last_reward", state.node_attrs["last_reward"] - penalty)
        return state.update_global_attr(
            "resource_level",
            state.global_attrs["resource_level"] + cfg.confiscate_rate * jnp.sum(over))
    return graduated_sanction
