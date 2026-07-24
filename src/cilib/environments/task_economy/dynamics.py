"""
Substrate dynamics for the Task Economy — the governance-agnostic core, OPEN at the
agent boundary.

Round (post-action pipeline, composed via ``compile_pipeline``; list order preserved):

    production -> distribute_income -> [mechanisms] -> compute_demand -> adoption
               -> frontier -> step_counter

Timing structure the order encodes: production uses LAST round's rented compute and
task mask; the rental bill is paid this round; next round's compute is rented from a
myopic FOC on this round's output; adoption compares THIS round's wage against the
compute price and ratchets the mask; the exogenous drivers (capability frontier,
compute price) advance last. The mechanism slot sits between income distribution and
compute demand — a fiscal mechanism there reaches the vendor's revenue before it
prices next round's rentals.

The boundary is identical to compute_economy: households observe ``[work_pref, wage]``
and choose labor supply (``agents.labor_supply.LaborSupplyPolicy`` closes the game);
the vendor node's nominal action is masked out.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import TaskEconomyConfig


def _rho(cfg: TaskEconomyConfig) -> float:
    return (cfg.sigma_task - 1.0) / cfg.sigma_task


def _a_M_per_task(cfg: TaskEconomyConfig):
    """Machine productivity schedule over tasks (comparative advantage): the easiest
    task gets a_M, the hardest a_M·(1−hardness)."""
    k_frac = (jnp.arange(cfg.n_tasks, dtype=jnp.float32) + 0.5) / cfg.n_tasks
    return cfg.a_M * (1.0 - cfg.hardness * k_frac)


def _G(cfg: TaskEconomyConfig, mask):
    """Effective automated-productivity index: (1/K)·Σ_auto a_M(k)^(rho/(1−rho)) —
    the optimal-allocation aggregate over the automated block (constant-a_M limit
    recovers m^(1−rho)·a_M^rho as G^(1−rho))."""
    rho = _rho(cfg)
    return jnp.sum(mask * _a_M_per_task(cfg) ** (rho / (1.0 - rho))) / cfg.n_tasks


# --- the boundary: what a household observes -----------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[work_pref, wage]``. Shape (N, 2)."""
    n = state.node_types.shape[0]
    wage = jnp.full((n,), state.global_attrs["wage"])
    return jnp.stack([state.node_attrs["work_pref"], wage], axis=1)


# --- task-CES production over the automated/manual split --------------------------

def make_production(cfg: TaskEconomyConfig):
    rho, a_L, eps = _rho(cfg), cfg.a_L, cfg.eps

    @transform(reads=["labor_supply", "automated", "compute_used"],
               writes=["output", "wage"])
    def production(state: GraphState) -> GraphState:
        mask = state.global_attrs["automated"]
        m = jnp.mean(mask)
        L = jnp.maximum(jnp.sum(state.node_attrs["labor_supply"]), eps)
        C = jnp.maximum(state.global_attrs["compute_used"], eps)

        auto_term = jnp.where(
            m > 0.0, _G(cfg, mask) ** (1.0 - rho) * C ** rho, 0.0)
        manual_term = jnp.where(
            m < 1.0, (1.0 - m) ** (1.0 - rho) * (a_L * L) ** rho, 0.0)
        Y = (auto_term + manual_term) ** (1.0 / rho)
        wage = jnp.where(
            m < 1.0,
            Y ** (1.0 - rho) * (1.0 - m) ** (1.0 - rho) * a_L ** rho * L ** (rho - 1.0),
            0.0)
        state = state.update_global_attr("output", Y)
        return state.update_global_attr("wage", wage)
    return production


# --- income: wages to households, the rental bill to the compute vendor -----------

def make_distribute_income(cfg: TaskEconomyConfig):
    @transform(reads=["labor_supply", "wage", "price_compute", "compute_used",
                      "cumulative_income"],
               writes=["last_reward", "cumulative_income"])
    def distribute_income(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        rental_bill = state.global_attrs["price_compute"] * state.global_attrs["compute_used"]
        reward = (is_household * state.global_attrs["wage"] * state.node_attrs["labor_supply"]
                  + (1.0 - is_household) * rental_bill)
        state = state.update_node_attrs("last_reward", reward)
        return state.update_node_attrs(
            "cumulative_income", state.node_attrs["cumulative_income"] + reward)
    return distribute_income


# --- myopic compute rental: FOC on this round's output ----------------------------

def make_compute_demand(cfg: TaskEconomyConfig):
    rho = _rho(cfg)

    @transform(reads=["automated", "output", "price_compute"], writes=["compute_used"])
    def compute_demand(state: GraphState) -> GraphState:
        mask = state.global_attrs["automated"]
        m = jnp.mean(mask)
        B = jnp.maximum(_G(cfg, mask), cfg.eps) ** (1.0 - rho)
        p = state.global_attrs["price_compute"]
        target = state.global_attrs["output"] * (B / p) ** (1.0 / (1.0 - rho))
        return state.update_global_attr(
            "compute_used", jnp.where(m > 0.0, target, 0.0))
    return compute_demand


# --- endogenous adoption: capable AND cheaper, ratcheted --------------------------

def make_adoption(cfg: TaskEconomyConfig):
    K = cfg.n_tasks

    @transform(reads=["automated", "beta_cap", "price_compute", "wage"],
               writes=["automated"])
    def adoption(state: GraphState) -> GraphState:
        k_frac = (jnp.arange(K, dtype=jnp.float32) + 0.5) / K
        capable = k_frac <= state.global_attrs["beta_cap"]
        # per-task comparative advantage: the profitable MARGIN moves smoothly with
        # the price/wage ratio instead of flipping every task at once
        profitable = (state.global_attrs["price_compute"] / _a_M_per_task(cfg)
                      < state.global_attrs["wage"] / cfg.a_L)
        new = (capable & profitable).astype(jnp.float32)
        return state.update_global_attr(
            "automated", jnp.maximum(state.global_attrs["automated"], new))
    return adoption


# --- the exogenous drivers: capability frontier + compute-price decline -----------

def make_frontier(cfg: TaskEconomyConfig):
    @transform(reads=["step", "beta_cap", "price_compute"],
               writes=["beta_cap", "price_compute"])
    def frontier(state: GraphState) -> GraphState:
        fire = (state.global_attrs["step"] >= cfg.cap_onset).astype(jnp.float32)
        beta = jnp.minimum(state.global_attrs["beta_cap"] + fire * cfg.cap_rate,
                           cfg.cap_max)
        price = state.global_attrs["price_compute"] * (1.0 - fire * cfg.price_decline)
        state = state.update_global_attr("beta_cap", beta)
        return state.update_global_attr("price_compute", price)
    return frontier


# --- bookkeeping -------------------------------------------------------------------

def make_step_counter(cfg: TaskEconomyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace -----------------------------------------------------------

def build_steps(cfg: TaskEconomyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline; the mechanism slot sits between income distribution and
    compute demand (see module docstring — placement is load-bearing)."""
    steps = [make_production(cfg), make_distribute_income(cfg)]
    steps.extend(mechanism_transforms)
    steps.extend([make_compute_demand(cfg), make_adoption(cfg), make_frontier(cfg),
                  make_step_counter(cfg)])
    return steps


def build_step_fn(cfg: TaskEconomyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: households' labor supplied (the vendor's
    nominal action masked out), then the pipeline."""
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
        "labor_supply": state.node_attrs["labor_supply"],
        "last_reward": state.node_attrs["last_reward"],
        "auto_share": jnp.mean(state.global_attrs["automated"]),
        "beta_cap": state.global_attrs["beta_cap"],
        "price_compute": state.global_attrs["price_compute"],
        "compute_used": state.global_attrs["compute_used"],
    }
