"""
Interventions family — the GD game's player-enacted defense cards
(docs/gd-game-design.md, decisions dated 2026-07-31).

One composed transform carries a branch's whole intervention plan. The cards
share write fields (``wealth``, ``enforcement``), so shipping them as separate
mechanisms would break the same-family disjoint-writes invariant — bundling is
the same move ``graduated_sanction`` makes for penalize + confiscate. Gating is
branchless (``jnp.where`` on the traced ``step`` global) rather than
``core.schedule.scheduled`` because a game branch is a *full run from t=0 with
a fixed plan* — the plan is static config, closed over, never state.

The four in-plan cards (the fifth card, the influence cap, is a channel
property and lives in ``ledger_society`` config as ``reach_cut``):

- **AI wealth levy** (economic): from ``levy_onset``, transfer
  ``levy_rate x wealth`` from AI actors equally to humans each tick. A
  conserving node->node transfer — total wealth is untouched.
- **Attention campaign** (economic): at exactly ``campaign_onset``, shift
  ``campaign_shift`` of each human's allocation preference from consume to
  broadcast. Persistent (``alloc_pref`` is the policy-read field); the cost is
  intrinsic — consumption foregone, spending routed to bought reach.
- **Fund repair** (economic -> political): from ``repair_onset``, drip
  ``repair_spend_rate x wealth`` from each human into the declared
  ``intervention_spend`` sink and convert the per-human spend into a bounded
  enforcement uplift. Money leaves the loop; capacity is bought.
- **Enforcement debits** (the political price tag): each ``(tick, cost)`` pair
  fires once, debiting the enforcement stock — enacting a political card
  spends institutional capacity. Affordability is judged by the tree
  generator at generation time; the debit here is the in-model cost.

With an empty plan (all onsets at ``NEVER``, no debits) the transform is a
bit-exact identity on every field it writes — the sealed-dial convention,
asserted as a test.

Prices and rates are typed on the game config (experiments/gd_game); none is
anchored — they are tuned-for-legibility, and the game's manifest says so.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform

NEVER = 10 ** 8   # an onset later than any run (the wp3_politics idiom)

PLAN_LEVERS = ("levy_rate", "repair_rate", "reach_cut", "sortition_rate")


@dataclasses.dataclass(frozen=True)
class InterventionPlanConfig:
    # card: AI wealth levy (conserving transfer, per tick from onset)
    levy_onset: int = NEVER
    levy_rate: float = 0.10
    # card: attention campaign (one-shot alloc_pref shift, consume -> broadcast)
    campaign_onset: int = NEVER
    campaign_shift: float = 0.10
    # card: fund repair (per-tick wealth drip -> intervention_spend sink,
    # bounded enforcement uplift)
    repair_onset: int = NEVER
    repair_spend_rate: float = 0.02
    repair_efficiency: float = 0.5
    # political enactment costs: ((tick, cost), ...), each fires exactly once
    enforcement_debits: Tuple[Tuple[int, float], ...] = ()


@dataclasses.dataclass(frozen=True)
class PolicyLeverConfig:
    """The live policy game's four levers (docs/remote-engine-design.md,
    2026-07-31). Ranges are the server's whitelist AND the in-transform clip;
    the upkeep coefficient is the continuous cost model — political intensity
    drains enforcement per tick instead of a one-shot debit. All values
    tuned-for-legibility; none anchored."""
    levy_max: float = 0.3
    repair_max: float = 0.05
    reach_cut_max: float = 1.0
    sortition_max: float = 0.2
    repair_efficiency: float = 0.5
    upkeep: float = 0.004    # enforcement drain per tick per unit of summed
                             # normalized political intensity


def make_policy_levers(cfg: PolicyLeverConfig = PolicyLeverConfig()):
    """Per-tick policy plan reader for the live game: the (T, 4) plan array in
    ``global_attrs["policy_plan"]`` is DATA (a dynamic pytree child), so one
    compiled program serves every plan. Column order: ``PLAN_LEVERS``. Every
    lever is exactly neutral at 0 (bit-identity tested). Not composable with
    ``make_interventions`` in the same pipeline — both write the same fields;
    the card game and the policy game are alternative closures."""

    @transform(reads=["step", "policy_plan", "wealth", "enforcement",
                      "intervention_spend", "delegation", "reach_cut_now"],
               writes=["wealth", "enforcement", "intervention_spend",
                       "delegation", "reach_cut_now"])
    def policy_levers(state: GraphState) -> GraphState:
        plan = state.global_attrs["policy_plan"]
        row = plan[jnp.clip(state.global_attrs["step"], 0, plan.shape[0] - 1)]
        levy = jnp.clip(row[0], 0.0, cfg.levy_max)
        repair = jnp.clip(row[1], 0.0, cfg.repair_max)
        cut = jnp.clip(row[2], 0.0, cfg.reach_cut_max)
        sort_r = jnp.clip(row[3], 0.0, cfg.sortition_max)

        is_human = state.node_types == 0
        h = is_human.astype(jnp.float32)
        n_h = jnp.maximum(jnp.sum(h), 1.0)

        # levy: conserving transfer, AI -> humans equally
        wealth = state.node_attrs["wealth"]
        take = levy * wealth * (1.0 - h)
        wealth = wealth - take + jnp.sum(take) / n_h * h

        # repair: drip into the declared sink, buy enforcement
        drip = repair * wealth * h
        wealth = wealth - drip
        enf = state.global_attrs["enforcement"]
        enf = enf + cfg.repair_efficiency * (jnp.sum(drip) / n_h) * (1.0 - enf)

        # upkeep: sustained political levers spend capacity continuously
        intensity = (levy / cfg.levy_max + cut / cfg.reach_cut_max
                     + sort_r / cfg.sortition_max)
        enf = jnp.clip(enf - cfg.upkeep * intensity, 0.0, 1.0)

        # sortition drip: blend citizen delegation rows toward the demos
        D = state.adj_matrices["delegation"]
        N = D.shape[0]
        eye = jnp.eye(N)
        lottery = h[None, :] * (1.0 - eye)
        lottery = lottery / jnp.maximum(
            jnp.sum(lottery, axis=1, keepdims=True), 1e-12)
        diag = jnp.diag(jnp.diag(D))
        off = D - diag
        blended = (1.0 - sort_r) * off + sort_r * jnp.sum(
            off, axis=1, keepdims=True) * lottery
        D_out = jnp.where(is_human[:, None], diag + blended, D)

        state = state.update_node_attrs("wealth", wealth)
        state = state.update_node_attrs("intervention_spend", drip)
        state = state.update_adj_matrix("delegation", D_out)
        state = state.update_global_attr("enforcement", enf)
        return state.update_global_attr("reach_cut_now", cut)
    return policy_levers


def make_interventions(cfg: InterventionPlanConfig):
    """The branch's intervention plan as one transform for the mechanism slot."""

    @transform(reads=["step", "wealth", "alloc_pref", "enforcement",
                      "intervention_spend"],
               writes=["wealth", "alloc_pref", "enforcement",
                       "intervention_spend"])
    def interventions(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        is_human = state.node_types == 0
        h = is_human.astype(jnp.float32)
        n_h = jnp.maximum(jnp.sum(h), 1.0)
        wealth = state.node_attrs["wealth"]

        # AI wealth levy: conserving transfer, AI -> humans equally
        on_levy = (step >= cfg.levy_onset).astype(jnp.float32)
        levy = on_levy * cfg.levy_rate * wealth * (1.0 - h)
        wealth = wealth - levy + jnp.sum(levy) / n_h * h

        # attention campaign: one-shot preference shift for humans
        pref = state.node_attrs["alloc_pref"]
        fire = step == cfg.campaign_onset
        move = jnp.where(fire & is_human,
                         jnp.minimum(cfg.campaign_shift, pref[:, 0]), 0.0)
        pref = pref.at[:, 0].add(-move).at[:, 2].add(move)

        # fund repair: drip into the declared sink, buy enforcement
        on_rep = (step >= cfg.repair_onset).astype(jnp.float32)
        drip = on_rep * cfg.repair_spend_rate * wealth * h
        wealth = wealth - drip
        spend_pc = jnp.sum(drip) / n_h
        enf = state.global_attrs["enforcement"]
        enf = enf + cfg.repair_efficiency * spend_pc * (1.0 - enf)

        # political enactment debits: each fires exactly once
        for tick, cost in cfg.enforcement_debits:
            enf = enf - jnp.where(step == tick, cost, 0.0)
        enf = jnp.clip(enf, 0.0, 1.0)

        state = state.update_node_attrs("wealth", wealth)
        state = state.update_node_attrs("alloc_pref", pref)
        state = state.update_node_attrs("intervention_spend", drip)
        return state.update_global_attr("enforcement", enf)
    return interventions
