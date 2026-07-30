"""
Substrate dynamics for the Capital Economy — WP1's referee-accepted rules, in
program order (``compile_pipeline`` derives the DAG from declared reads/writes):

    spend -> arrive -> rebalance -> distribute -> [mechanism slot] -> accumulate
        -> grow -> step

Lag structure mirrors ``io_economy``: ``spend`` reads LAST tick's income and
wealth; ``rebalance`` reads LAST tick's capital-linked demand (``demand_k``,
written by ``accumulate``). The mechanism slot sits between income distribution
and accumulation, so a profit tax reduces what gets reinvested — the placement
that makes the eradication band (WP1 Prop. 4) reachable.

The referee-hardened rules this file implements exactly (main.tex §3.2):
- **Upkeep settlement**: sectors receive ``min(rev, m·K)`` — upkeep is paid out
  of revenue only; the unpaid remainder is *physical* charge-to-stock, not money.
- **Charge-to-stock**: when ``π < 0``, ``K ← [K + π]₊`` before depreciation.
- **Money/title separation, exactly once**: the owner pays the whole investment
  purchase (demand on the machine sector); title splits ``(1−ω, ω)``.
- **The r-closure**: discretionary outlays (investment, consumption) scale by
  ``r``; the unspent surplus parks in the tracked hoard — a demand stall, money
  conserved at every ``r`` (the conservation probe checks this on every run).
- **Public fund**: earns pro rata, pays upkeep by the same min rule, pays its
  positive profit to households as an equal dividend, never reinvests; its stock
  grows only through the ``ω`` slice of private investment.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import CapitalEconomyConfig


def observe_fn(state: GraphState):
    """Per-agent observation: the agent's own spend-preference row. Shape (N, S)."""
    return state.node_attrs["spend_pref"]


def _sector_onehot(cfg: CapitalEconomyConfig, home_sector):
    """(N, N) one-hot from each owner row to its home-sector column."""
    n = home_sector.shape[0]
    return (jnp.arange(n)[None, :] == home_sector[:, None]).astype(jnp.float32)


# --- household spending: last income + wealth drawdown becomes final demand ------

def make_spend(cfg: CapitalEconomyConfig):
    H = cfg.n_households

    @transform(reads=["last_reward", "wealth", "spend_weights"], writes=["demand_h"])
    def spend(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        budget = ((1.0 - cfg.sigma_s) * jnp.maximum(state.node_attrs["last_reward"], 0.0)
                  + cfg.sigma_d * state.node_attrs["wealth"]) * is_household
        d_sector = state.node_attrs["spend_weights"].T @ budget          # (S,)
        demand_h = jnp.zeros_like(budget).at[H:H + cfg.n_sectors].set(d_sector)
        return state.update_node_attrs("demand_h", demand_h)
    return spend


# --- arrivals: dormant owners activate on schedule with their seed capital --------

def make_arrive(cfg: CapitalEconomyConfig):
    @transform(reads=["step", "arrival_tick", "active", "capital", "pub_cap",
                      "home_sector"],
               writes=["active", "capital", "pub_cap"])
    def arrive(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        is_owner = (state.node_types == 2).astype(jnp.float32)
        newly = is_owner * (step == state.node_attrs["arrival_tick"]).astype(jnp.float32)
        active = jnp.maximum(state.node_attrs["active"], newly)
        # seed capital splits (1-omega, omega) at arrival, like any investment
        k_seed = newly * cfg.init_capital * (1.0 - cfg.ownership)
        pub_seed = (_sector_onehot(cfg, state.node_attrs["home_sector"]).T
                    @ (newly * cfg.init_capital * cfg.ownership))
        state = state.update_node_attrs("active", active)
        state = state.update_node_attrs("capital", state.node_attrs["capital"] + k_seed)
        return state.update_node_attrs("pub_cap", state.node_attrs["pub_cap"] + pub_seed)
    return arrive


# --- output rebalancing: one Neumann step toward the Leontief solution ------------

def make_rebalance(cfg: CapitalEconomyConfig):
    @transform(reads=["technical", "gross_output", "demand_h", "demand_k"],
               writes=["gross_output"])
    def rebalance(state: GraphState) -> GraphState:
        x = state.node_attrs["gross_output"]
        d = state.node_attrs["demand_h"] + state.node_attrs["demand_k"]
        x_next = state.adj_matrices["technical"] @ x + d
        return state.update_node_attrs("gross_output", jnp.maximum(x_next, 0.0))
    return rebalance


# --- income: value added splits by automation share; upkeep settled by min --------

def make_distribute(cfg: CapitalEconomyConfig):
    H, S = cfg.n_households, cfg.n_sectors

    @transform(reads=["gross_output", "technical", "capital", "pub_cap", "active",
                      "home_sector", "wealth", "efficiency"],
               writes=["last_reward", "capital_income", "capital", "pub_cap",
                       "upkeep_paid"])
    def distribute(state: GraphState) -> GraphState:
        x = state.node_attrs["gross_output"]
        A = state.adj_matrices["technical"]
        is_household = (state.node_types == 0).astype(jnp.float32)
        is_sector = (state.node_types == 1).astype(jnp.float32)
        onehot = _sector_onehot(cfg, state.node_attrs["home_sector"])    # (N, N)

        va = jnp.maximum(1.0 - jnp.sum(A, axis=0), 0.0) * x * is_sector  # v_j x_j

        K_own = state.node_attrs["capital"] * state.node_attrs["active"]
        sec_priv = onehot.T @ K_own                                      # (N,) sector slots
        sec_tot = sec_priv + state.node_attrs["pub_cap"]
        capacity = state.global_attrs["efficiency"] * sec_tot
        a_share = capacity / (capacity + 1.0) * is_sector                # a_j

        labor_pay = jnp.sum((1.0 - a_share) * va) / H
        cap_pay = a_share * va                                           # (N,) sector slots

        # pro-rata revenue; upkeep settled min(rev, mK); loss charged to stock
        share_priv = jnp.where(sec_tot > cfg.eps, cap_pay / jnp.maximum(sec_tot, cfg.eps), 0.0)
        rev_own = (onehot @ share_priv) * K_own                          # owners
        rev_pub = share_priv * state.node_attrs["pub_cap"]               # sectors
        upkeep_own = jnp.minimum(rev_own, cfg.maintenance * K_own)
        upkeep_pub = jnp.minimum(rev_pub, cfg.maintenance * state.node_attrs["pub_cap"])
        pi_own = rev_own - cfg.maintenance * K_own
        pi_pub = rev_pub - cfg.maintenance * state.node_attrs["pub_cap"]

        capital = jnp.maximum(state.node_attrs["capital"] + jnp.minimum(pi_own, 0.0), 0.0)
        pub_cap = jnp.maximum(state.node_attrs["pub_cap"] + jnp.minimum(pi_pub, 0.0), 0.0)

        # fund disposal (config.pub_mirror, static): the dividend fund pays all
        # profit to households; the mirror fund retains the s-slice (and the
        # ratchet follows — see config.py and WP1 Prop. 4's measured regimes).
        pub_profit = jnp.maximum(pi_pub, 0.0)
        payout_share = (1.0 - cfg.reinvest_rate) if cfg.pub_mirror else 1.0
        dividend = payout_share * jnp.sum(pub_profit) / H
        # owners' money lives in capital_income -> hoard; last_reward stays a
        # household stock (the conservation probe counts money exactly once)
        reward = is_household * (labor_pay + dividend)
        state = state.update_node_attrs("last_reward", reward)
        state = state.update_node_attrs("capital_income", jnp.maximum(pi_own, 0.0))
        state = state.update_node_attrs("pub_profit", pub_profit)
        state = state.update_node_attrs("capital", capital)
        state = state.update_node_attrs("pub_cap", pub_cap)
        return state.update_global_attr(
            "upkeep_paid", jnp.sum(upkeep_own) + jnp.sum(upkeep_pub))
    return distribute


# --- accumulation: reinvest (title split), hoard, consume, route capital demand ---

def make_accumulate(cfg: CapitalEconomyConfig):
    H, S = cfg.n_households, cfg.n_sectors

    @transform(reads=["capital_income", "capital", "pub_cap", "pub_profit",
                      "wealth", "active", "home_sector", "last_reward",
                      "upkeep_paid"],
               writes=["capital", "pub_cap", "wealth", "demand_k"])
    def accumulate(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        is_owner = (state.node_types == 2).astype(jnp.float32)
        onehot = _sector_onehot(cfg, state.node_attrs["home_sector"])
        act = state.node_attrs["active"]

        # post-mechanism profit base (the tax already reduced capital_income here)
        profit = state.node_attrs["capital_income"] * is_owner * act
        invest_budget = cfg.reinvest_rate * profit
        invest_spent = cfg.recycle * invest_budget                       # r-closure
        consume = cfg.recycle * cfg.consume_rate * state.node_attrs["wealth"] * is_owner

        # capital formation follows spending; title splits (1-omega, omega).
        # Mirror fund reinvests its retained s-slice (always full-recycle — an
        # institution, not a discretionary spender); dividend fund retains none.
        pub_s = cfg.reinvest_rate if cfg.pub_mirror else 0.0
        pub_reinvest = pub_s * state.node_attrs["pub_profit"]
        capital = state.node_attrs["capital"] * (1.0 - cfg.depreciation) \
            + (1.0 - cfg.ownership) * invest_spent
        pub_cap = state.node_attrs["pub_cap"] * (1.0 - cfg.depreciation) \
            + (onehot.T @ (cfg.ownership * invest_spent)) + pub_reinvest

        # hoards: retained profit + the stalled (unspent) surplus - consumption
        wealth_own = jnp.maximum(
            state.node_attrs["wealth"] * is_owner
            + (profit - invest_spent) - consume, 0.0)
        wealth_hh = jnp.maximum(
            state.node_attrs["wealth"] * is_household
            + cfg.sigma_s * jnp.maximum(state.node_attrs["last_reward"], 0.0) * is_household
            - cfg.sigma_d * state.node_attrs["wealth"] * is_household, 0.0)
        wealth = wealth_own + wealth_hh

        # capital-linked demand, landing next tick: upkeep + all investment
        # (private and fund) on machines, AI consumption spread evenly
        machines = jnp.zeros_like(profit).at[H].set(1.0)
        even = jnp.zeros_like(profit).at[H:H + S].set(1.0 / S)
        demand_k = machines * (state.global_attrs["upkeep_paid"]
                               + jnp.sum(invest_spent) + jnp.sum(pub_reinvest)) \
            + even * jnp.sum(consume)

        state = state.update_node_attrs("capital", capital)
        state = state.update_node_attrs("pub_cap", pub_cap)
        state = state.update_node_attrs("wealth", wealth)
        return state.update_node_attrs("demand_k", demand_k)
    return accumulate


# --- capability growth: e <- min(cap, e·(1 + g + γ·e)), from first arrival --------
# The initial ``efficiency`` is capability AT DEPLOYMENT; growth runs once AI
# capital exists (gated by step, jnp.where — no data-dependent control flow).

def make_grow(cfg: CapitalEconomyConfig):
    @transform(reads=["efficiency", "step"], writes=["efficiency"])
    def grow(state: GraphState) -> GraphState:
        e = state.global_attrs["efficiency"]
        grown = jnp.minimum(e * (1.0 + cfg.growth_rate + cfg.rsi_strength * e),
                            cfg.e_ceiling)
        live = (state.global_attrs["step"] >= cfg.first_arrival).astype(jnp.float32)
        return state.update_global_attr("efficiency", live * grown + (1.0 - live) * e)
    return grow


# --- bookkeeping ------------------------------------------------------------------

def make_step_counter(cfg: CapitalEconomyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace -----------------------------------------------------------

def build_steps(cfg: CapitalEconomyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline; the mechanism slot sits between income distribution and
    accumulation (a profit tax there shrinks reinvestment — WP1 Prop. 4)."""
    steps = [make_spend(cfg), make_arrive(cfg), make_rebalance(cfg),
             make_distribute(cfg)]
    steps.extend(mechanism_transforms)
    steps.extend([make_accumulate(cfg), make_grow(cfg), make_step_counter(cfg)])
    return steps


def build_step_fn(cfg: CapitalEconomyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: households' spending weights normalized
    (non-household rows masked), then the compiled pipeline."""
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
    """Raw per-step readouts (N ≈ 32 — small enough to keep; reduction in metrics)."""
    return {
        "gross_output": state.node_attrs["gross_output"],
        "capital": state.node_attrs["capital"],
        "pub_cap": state.node_attrs["pub_cap"],
        "wealth": state.node_attrs["wealth"],
        "last_reward": state.node_attrs["last_reward"],
        "demand_h": state.node_attrs["demand_h"],
        "demand_k": state.node_attrs["demand_k"],
        "active": state.node_attrs["active"],
        "efficiency": state.global_attrs["efficiency"],
    }
