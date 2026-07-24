"""
Substrate dynamics for the Compute Economy — the governance-agnostic core, OPEN at the
agent boundary.

Round (post-action pipeline, composed via ``compile_pipeline``):

    arrival -> production -> distribute_income -> [mechanisms] -> reinvest -> step_counter

The mechanism slot sits **between income distribution and reinvestment** — deliberately:
a fiscal mechanism that taxes ``capital_income`` there slows compute compounding, which
is what bends the labor-share curve. Spliced after reinvestment it would be a silent
no-op on the dynamics (verified numerically in the calibration prototype).

The boundary: households observe ``[work_pref, wage]`` and choose labor supply
(``agents.labor_supply.LaborSupplyPolicy`` is the default closure); AI actors' nominal
actions are masked out in ``step_fn`` — their behavior is the fixed reinvestment rule.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import ComputeEconomyConfig


# --- the boundary: what a household observes -----------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[work_pref, wage]``. Shape (N, 2)."""
    n = state.node_types.shape[0]
    wage = jnp.full((n,), state.global_attrs["wage"])
    return jnp.stack([state.node_attrs["work_pref"], wage], axis=1)


# --- scheduled AI arrivals (boundary rule: pre-allocated slots, monotonic mask) --

def make_arrival(cfg: ComputeEconomyConfig):
    arrival_ticks = jnp.concatenate([
        jnp.full(cfg.n_households, -1, dtype=jnp.int32),        # households: active from t=0
        cfg.first_arrival_tick
        + cfg.arrival_spacing * jnp.arange(cfg.n_ai_slots, dtype=jnp.int32),
    ])

    @transform(reads=["step", "active", "capital"], writes=["active", "capital"])
    def arrival(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        eligible = step >= arrival_ticks
        newly = eligible & (state.node_attrs["active"] < 0.5)
        active = jnp.maximum(state.node_attrs["active"], eligible.astype(jnp.float32))
        capital = jnp.where(newly, cfg.initial_ai_capital, state.node_attrs["capital"])
        state = state.update_node_attrs("active", active)
        return state.update_node_attrs("capital", capital)
    return arrival


# --- CES production + factor prices ----------------------------------------------

def make_production(cfg: ComputeEconomyConfig):
    """Y, wage (MPL), return to compute (MPC). ``rho`` is static config, so the
    Cobb-Douglas limit is a factory-build-time Python branch — not traced control flow."""
    a, A, eps = cfg.alpha, cfg.A, cfg.eps

    if abs(cfg.rho) < 1e-8:                      # Cobb-Douglas limit: Y = A·L^α·C^(1−α)
        def factor_prices(L, C):
            L_s, C_s = jnp.maximum(L, eps), jnp.maximum(C, eps)
            Y = A * L_s ** a * C_s ** (1.0 - a)
            wage = a * Y / L_s
            r = jnp.where(C > eps, (1.0 - a) * Y / C_s, 0.0)
            return Y, wage, r
    else:
        rho = cfg.rho

        def factor_prices(L, C):
            L_s, C_s = jnp.maximum(L, eps), jnp.maximum(C, eps)
            inner = a * L_s ** rho + (1.0 - a) * C ** rho
            Y = A * inner ** (1.0 / rho)
            wage = a * A ** rho * (Y / L_s) ** (1.0 - rho)
            r = jnp.where(C > eps, (1.0 - a) * A ** rho * (Y / C_s) ** (1.0 - rho), 0.0)
            return Y, wage, r

    @transform(reads=["labor_supply", "capital", "active"],
               writes=["output", "wage", "return_to_compute"])
    def production(state: GraphState) -> GraphState:
        L = jnp.sum(state.node_attrs["labor_supply"])
        C = jnp.sum(state.node_attrs["capital"] * state.node_attrs["active"])
        Y, wage, r = factor_prices(L, C)
        state = state.update_global_attr("output", Y)
        state = state.update_global_attr("wage", wage)
        return state.update_global_attr("return_to_compute", r)
    return production


# --- income distribution (wage·L + r·C = Y exactly, by the CES Euler identity) ----

def make_distribute_income(cfg: ComputeEconomyConfig):
    @transform(reads=["labor_supply", "capital", "active", "wage", "return_to_compute"],
               writes=["last_reward", "capital_income"])
    def distribute_income(state: GraphState) -> GraphState:
        labor_income = state.global_attrs["wage"] * state.node_attrs["labor_supply"]
        capital_income = (state.global_attrs["return_to_compute"]
                          * state.node_attrs["capital"] * state.node_attrs["active"])
        state = state.update_node_attrs("capital_income", capital_income)
        return state.update_node_attrs("last_reward", labor_income + capital_income)
    return distribute_income


# --- compute compounding (reads possibly-taxed capital_income) --------------------

def make_reinvest(cfg: ComputeEconomyConfig):
    @transform(reads=["last_reward", "capital_income", "capital", "cumulative_income"],
               writes=["capital", "cumulative_income"])
    def reinvest(state: GraphState) -> GraphState:
        capital = (state.node_attrs["capital"] * (1.0 - cfg.depreciation)
                   + cfg.reinvest_rate * state.node_attrs["capital_income"])
        state = state.update_node_attrs("capital", capital)
        return state.update_node_attrs(
            "cumulative_income",
            state.node_attrs["cumulative_income"] + state.node_attrs["last_reward"])
    return reinvest


# --- bookkeeping -------------------------------------------------------------------

def make_step_counter(cfg: ComputeEconomyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace --------------------------------------------------------------

def build_steps(cfg: ComputeEconomyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline; the mechanism slot sits between income distribution and
    reinvestment (see module docstring — placement is load-bearing)."""
    steps = [make_arrival(cfg), make_production(cfg), make_distribute_income(cfg)]
    steps.extend(mechanism_transforms)
    steps.extend([make_reinvest(cfg), make_step_counter(cfg)])
    return steps


def build_step_fn(cfg: ComputeEconomyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: households' labor supplied (AI slots masked
    out — their behavior is the reinvestment rule, not an action), then the pipeline."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        is_household = (state.node_types == 0).astype(jnp.float32)
        state = state.update_node_attrs(
            "labor_supply", jnp.maximum(actions, 0.0) * is_household)
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Raw per-step readouts; reduction happens in metrics.py / the harness."""
    return {
        "output": state.global_attrs["output"],
        "wage": state.global_attrs["wage"],
        "return_to_compute": state.global_attrs["return_to_compute"],
        "labor_supply": state.node_attrs["labor_supply"],
        "capital": state.node_attrs["capital"],
        "last_reward": state.node_attrs["last_reward"],
        "active": state.node_attrs["active"],
    }
