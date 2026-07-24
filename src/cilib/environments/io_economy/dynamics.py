"""
Substrate dynamics for the IO Economy — the governance-agnostic core, OPEN at the
agent boundary.

Round (post-action pipeline, composed via ``compile_pipeline``; list order is the
program order the compiler preserves):

    spend -> substitute -> rebalance -> distribute -> [mechanisms] -> ai_demand -> step

The lag structure is deliberate: ``spend`` reads LAST round's income (it precedes
``distribute`` in program order, and the WAR hazard keeps it there), and ``rebalance``
reads LAST round's AI demand (``ai_demand`` comes after). The mechanism slot sits
between income distribution and AI demand formation — a fiscal mechanism that taxes
``capital_income`` there redirects AI surplus into household transfers BEFORE it
becomes self-directed AI demand, which is what defends the human demand share.

The boundary: households observe their own spend-preference row and choose spending
weights (``agents.spend_share.SpendSharePolicy`` is the default closure); sector rows
are masked out in ``step_fn`` — sectors don't decide, they balance.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import IOEconomyConfig


# --- the boundary: what a household observes -----------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: the agent's own spend-preference row. Shape (N, S)."""
    return state.node_attrs["spend_pref"]


# --- household spending: last round's income becomes this round's final demand --

def make_spend(cfg: IOEconomyConfig):
    H = cfg.n_households

    @transform(reads=["last_reward", "spend_weights"], writes=["demand_h"])
    def spend(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        budget = state.node_attrs["last_reward"] * is_household
        d_sector = state.node_attrs["spend_weights"].T @ budget          # (S,)
        demand_h = jnp.zeros_like(budget).at[H:].set(d_sector)
        return state.update_node_attrs("demand_h", demand_h)
    return spend


# --- AI substitution: recipes rewire, labor -> purchased cognition (cost parity) --

def make_substitute(cfg: IOEconomyConfig):
    @transform(reads=["step", "labor_coeff", "technical"],
               writes=["labor_coeff", "technical"])
    def substitute(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        rate = jnp.where(step >= cfg.sub_onset, cfg.sub_rate, 0.0)
        delta = rate * state.node_attrs["labor_coeff"]     # (N,), only ordinary > 0
        A = state.adj_matrices["technical"]
        n = A.shape[0]
        state = state.update_node_attrs(
            "labor_coeff", state.node_attrs["labor_coeff"] - delta)
        return state.update_adj_matrix("technical", A.at[n - 1, :].add(delta))
    return substitute


# --- output rebalancing: one Neumann step toward the Leontief solution ------------

def make_rebalance(cfg: IOEconomyConfig):
    @transform(reads=["technical", "gross_output", "demand_h", "demand_ai"],
               writes=["gross_output"])
    def rebalance(state: GraphState) -> GraphState:
        x = state.node_attrs["gross_output"]
        d = state.node_attrs["demand_h"] + state.node_attrs["demand_ai"]
        x_next = state.adj_matrices["technical"] @ x + d
        return state.update_node_attrs("gross_output", jnp.maximum(x_next, 0.0))
    return rebalance


# --- income: wage bill to households, margins to whoever owns them ----------------

def make_distribute_income(cfg: IOEconomyConfig):
    H = cfg.n_households

    @transform(reads=["gross_output", "labor_coeff", "technical"],
               writes=["last_reward", "capital_income", "wage_bill"])
    def distribute_income(state: GraphState) -> GraphState:
        x = state.node_attrs["gross_output"]
        l = state.node_attrs["labor_coeff"]
        is_household = (state.node_types == 0).astype(jnp.float32)
        is_sector = 1.0 - is_household

        wage_bill = jnp.sum(l * x)
        colsum = jnp.sum(state.adj_matrices["technical"], axis=0)
        margin = jnp.maximum(1.0 - colsum - l, 0.0) * is_sector
        state = state.update_node_attrs("capital_income", margin * x)
        state = state.update_node_attrs("last_reward", is_household * wage_bill / H)
        return state.update_global_attr("wage_bill", wage_bill)
    return distribute_income


# --- the AI loop: (possibly taxed) surplus becomes self-directed demand -----------

def make_ai_demand(cfg: IOEconomyConfig):
    @transform(reads=["capital_income"], writes=["demand_ai"])
    def ai_demand(state: GraphState) -> GraphState:
        n = state.node_types.shape[0]
        d = jnp.zeros(n, dtype=jnp.float32).at[n - 1].set(
            cfg.reinvest_rate * state.node_attrs["capital_income"][n - 1])
        return state.update_node_attrs("demand_ai", d)
    return ai_demand


# --- bookkeeping -------------------------------------------------------------------

def make_step_counter(cfg: IOEconomyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace -----------------------------------------------------------

def build_steps(cfg: IOEconomyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline; the mechanism slot sits between income distribution and
    AI demand formation (see module docstring — placement is load-bearing)."""
    steps = [make_spend(cfg), make_substitute(cfg), make_rebalance(cfg),
             make_distribute_income(cfg)]
    steps.extend(mechanism_transforms)
    steps.extend([make_ai_demand(cfg), make_step_counter(cfg)])
    return steps


def build_step_fn(cfg: IOEconomyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: households' spending weights normalized and
    applied (sector rows masked — sectors balance, they don't decide), then the
    pipeline."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        is_household = (state.node_types == 0).astype(jnp.float32)
        w = jnp.maximum(actions, 0.0) * is_household[:, None]
        w = w / (jnp.sum(w, axis=1, keepdims=True) + cfg.eps)
        state = state.update_node_attrs("spend_weights", w)
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Raw per-step readouts; reduction happens in metrics.py / the harness."""
    return {
        "gross_output": state.node_attrs["gross_output"],
        "labor_coeff": state.node_attrs["labor_coeff"],
        "technical": state.adj_matrices["technical"],
        "demand_h": state.node_attrs["demand_h"],
        "demand_ai": state.node_attrs["demand_ai"],
        "last_reward": state.node_attrs["last_reward"],
        "capital_income": state.node_attrs["capital_income"],
        "wage_bill": state.global_attrs["wage_bill"],
    }
