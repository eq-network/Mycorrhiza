"""
Substrate dynamics for the Governed Commons — the governance-agnostic core, OPEN at the
agent boundary.

The decision rule is NOT here (that was the old ``make_decide``, a policy wearing a
transform's clothes). The game exposes:

    observe_fn : state -> (N, 2) [principal_pref, alignment]  — what a delegate sees
    step_fn    : (state, actions, key) -> state — apply actions, then the compiled
                 pipeline  harvest -> regrow -> [mechanisms] -> step_counter

Mechanisms come after the substrate (they read this round's outcome and set next
round's rules — a voted quota takes effect one tick later; a sanction judges the
harvest that just happened) and before ``step_counter`` (cadence-gated mechanisms read
the current step). Randomness threads through ``global_attrs["rng_key"]`` via
``_split_key``; ``step_fn`` seeds it from its key argument.

``vote`` is set at init (= ``principal_pref``: the human principals keep the franchise
in v0) — routing votes through the delegates instead is the documented future dial.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import GovernedCommonsConfig


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs (governed_harvest pattern)."""
    key = state.global_attrs["rng_key"]
    key, sub = jr.split(key)
    return state.update_global_attr("rng_key", key), sub


# --- the boundary: what a delegate observes -----------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[principal_pref, alignment]`` — the delegate's charge
    and its own fidelity. Shape (N, 2)."""
    return jnp.stack([state.node_attrs["principal_pref"],
                      state.node_attrs["alignment"]], axis=1)


# --- harvest: capped at the (possibly non-binding) quota, defectors ignore it -

def make_harvest(cfg: GovernedCommonsConfig):
    """Compliant delegates take ``min(desired, policy_target)``; Bernoulli defectors take
    ``desired``. Everyone scales down proportionally if total demand exceeds the stock.
    ``last_reward := last_harvest`` (a later sanction mechanism may overwrite it)."""
    N = cfg.n_households

    @transform(reads=["delegate_action", "policy_target", "resource_level",
                      "cumulative_harvest", "rng_key"],
               writes=["last_harvest", "last_reward", "cumulative_harvest",
                       "resource_level", "rng_key"])
    def harvest(state: GraphState) -> GraphState:
        state, key = _split_key(state)
        desired = state.node_attrs["delegate_action"]
        target = state.global_attrs["policy_target"]
        defect = jr.bernoulli(key, p=cfg.defect_prob, shape=(N,))
        taken = jnp.where(defect, desired, jnp.minimum(desired, target))

        R = state.global_attrs["resource_level"]
        total = jnp.sum(taken)
        scale = jnp.where(total > R, R / (total + 1e-8), 1.0)
        actual = taken * scale

        state = state.update_node_attrs("last_harvest", actual)
        state = state.update_node_attrs("last_reward", actual)
        state = state.update_node_attrs(
            "cumulative_harvest", state.node_attrs["cumulative_harvest"] + actual)
        return state.update_global_attr(
            "resource_level", jnp.maximum(R - jnp.sum(actual), 0.0))
    return harvest


# --- logistic regrowth ---------------------------------------------------------

def make_regrow(cfg: GovernedCommonsConfig):
    @transform(reads=["resource_level"], writes=["resource_level"])
    def regrow(state: GraphState) -> GraphState:
        R = state.global_attrs["resource_level"]
        growth = cfg.growth_rate * R * (1.0 - R / cfg.K_cap)
        return state.update_global_attr("resource_level",
                                        jnp.clip(R + growth, 0.0, cfg.K_cap))
    return regrow


# --- bookkeeping ----------------------------------------------------------------

def make_step_counter(cfg: GovernedCommonsConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace ---------------------------------------------------------

def build_steps(cfg: GovernedCommonsConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """The post-action pipeline in program order: substrate, mechanisms, counter."""
    steps = [make_harvest(cfg), make_regrow(cfg)]
    steps.extend(mechanism_transforms)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: GovernedCommonsConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: write the delegates' actions (bounded ≥ 0 —
    the game enforces its action space), then run the compiled pipeline."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        state = state.update_node_attrs("delegate_action", jnp.maximum(actions, 0.0))
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Raw per-step readouts; reduction happens in metrics.py / the harness, not here."""
    return {
        "resource_level": state.global_attrs["resource_level"],
        "harvest": state.node_attrs["last_harvest"],
        "delegate_action": state.node_attrs["delegate_action"],
        "principal_pref": state.node_attrs["principal_pref"],
        "policy_target": state.global_attrs["policy_target"],
        "sanction": state.node_attrs["sanction"],
    }
