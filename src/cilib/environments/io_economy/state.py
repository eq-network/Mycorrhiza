"""
Initial-state factory for the IO Economy substrate.

Fixed-shape population: ``N = n_households + n_sectors``. ``node_types``: 0 = household
(indices ``[0, H)``), 1 = ordinary sector, 2 = the AI-cognition sector (index ``N−1``).
The technical-coefficients matrix lives where a graph library should keep it — in
``adj_matrices["technical"]`` (entry ``[i, j]`` = input from node i per unit of node
j's output; household rows/columns are zero, so the sector block is the IO matrix).

The economy starts AT its no-AI steady state: household preferences draw the initial
final-demand vector, and ``gross_output`` is initialized to the exact Leontief solution
``x₀ = (I−A)⁻¹ d₀`` — so the pre-onset trajectory is stationary by construction and
the rebalancing dynamics' convergence is testable against a known fixed point.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import IOEconomyConfig


def make_state(cfg: IOEconomyConfig, key) -> GraphState:
    H, S = cfg.n_households, cfg.n_sectors
    N = H + S
    n_ord = S - 1                                   # ordinary sectors; AI is index N-1

    # recipes: ordinary columns buy a_intra from every ordinary sector; the AI column
    # buys a_ai_inputs from every ordinary sector; the AI row starts empty (nobody has
    # replaced labor with cognition yet).
    A = jnp.zeros((N, N), dtype=jnp.float32)
    ord_idx = jnp.arange(H, H + n_ord)
    A = A.at[jnp.ix_(ord_idx, ord_idx)].set(cfg.a_intra)
    A = A.at[ord_idx, N - 1].set(cfg.a_ai_inputs)

    # labor coefficients: zero-margin ordinary sectors (l = 1 − column sum); the AI
    # sector employs no labor — its value added is its margin.
    labor_coeff = jnp.zeros(N, dtype=jnp.float32)
    labor_coeff = labor_coeff.at[ord_idx].set(1.0 - cfg.a_intra * n_ord)

    # household spend preferences over sectors (columns = sector offsets [0, S); the
    # AI column stays 0 — cognition is an intermediate good, not a consumption good).
    base = jnp.full((H, n_ord), 1.0 / n_ord)
    noise = cfg.pref_noise * jr.normal(key, (H, n_ord))
    pref_ord = jnp.maximum(base + noise, 0.01)
    pref_ord = pref_ord / jnp.sum(pref_ord, axis=1, keepdims=True)
    spend_pref = jnp.zeros((N, S), dtype=jnp.float32)
    spend_pref = spend_pref.at[:H, :n_ord].set(pref_ord)

    # bootstrap demand and the exact steady-state output it implies
    income0 = jnp.concatenate([jnp.full(H, cfg.init_income), jnp.zeros(S)])
    d0 = jnp.zeros(N, dtype=jnp.float32)
    d0 = d0.at[H:].set(spend_pref[:H].T @ income0[:H])
    x0 = jnp.linalg.solve(jnp.eye(N) - A, d0)

    node_types = jnp.concatenate([
        jnp.zeros(H, dtype=jnp.int32),
        jnp.ones(n_ord, dtype=jnp.int32),
        jnp.full((1,), 2, dtype=jnp.int32),
    ])
    node_attrs = {
        "gross_output": x0.astype(jnp.float32),
        "labor_coeff": labor_coeff,
        "spend_pref": spend_pref,
        "spend_weights": jnp.zeros((N, S), dtype=jnp.float32),
        "demand_h": d0,
        "demand_ai": jnp.zeros(N, dtype=jnp.float32),
        "last_reward": income0.astype(jnp.float32),
        "capital_income": jnp.zeros(N, dtype=jnp.float32),
        "active": jnp.ones(N, dtype=jnp.float32),
    }
    global_attrs = {
        "wage_bill": jnp.array(H * cfg.init_income, dtype=jnp.float32),
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"technical": A},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: IOEconomyConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
