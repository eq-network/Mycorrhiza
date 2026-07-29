"""
Initial-state factory for the Capital Economy substrate.

Fixed-shape population ``N = n_households + n_sectors + n_owners``. ``node_types``:
0 = household ``[0, H)``, 1 = sector ``[H, H+S)`` (index ``H`` is the machine
sector), 2 = AI owner ``[H+S, N)`` (owner ``i``'s home sector is ``i % S``).

The economy starts AT its households-only steady state: ``gross_output`` is the
exact Leontief solution for the initial human demand, household wealth sits at the
conserved-loop fixed ratio ``sigma_s/sigma_d`` (WP1 Prop. 2), and every owner is
dormant (``active = 0``, zero capital) until its scheduled arrival tick. The
pre-arrival trajectory is therefore stationary by construction — the
limit-equivalence rung tests exactly this.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import CapitalEconomyConfig


def technical_matrix(cfg: CapitalEconomyConfig):
    """The (static) hub+chain recipe matrix — shared by init and by the money
    invariant (inventories-in-process = 1ᵀAx are a tracked SFC stock)."""
    H, S = cfg.n_households, cfg.n_sectors
    N = H + S + cfg.n_owners
    sec = jnp.arange(H, H + S)
    A = jnp.zeros((N, N), dtype=jnp.float32)
    A = A.at[H, sec[1:]].set(cfg.a_machines)                  # machines feed everyone
    A = A.at[sec[1:-1], sec[2:]].set(cfg.a_chain)             # chain j-1 -> j (j >= 2)
    return A


def make_state(cfg: CapitalEconomyConfig, key) -> GraphState:
    H, S, O = cfg.n_households, cfg.n_sectors, cfg.n_owners
    N = H + S + O
    A = technical_matrix(cfg)

    # household spend preferences over the S sectors
    base = jnp.full((H, S), 1.0 / S)
    noise = cfg.pref_noise * jr.normal(key, (H, S))
    pref = jnp.maximum(base + noise, 0.01)
    pref = pref / jnp.sum(pref, axis=1, keepdims=True)
    spend_pref = jnp.zeros((N, S), dtype=jnp.float32).at[:H].set(pref)

    # bootstrap demand and its exact Leontief output. At the conserved-loop fixed
    # point, spend = (1-sigma_s)*y + sigma_d*W* = y exactly (W*/y = sigma_s/sigma_d).
    income0 = jnp.zeros(N, dtype=jnp.float32).at[:H].set(cfg.init_income)
    d_sector = spend_pref[:H].T @ jnp.full(H, cfg.init_income)          # (S,)
    d0 = jnp.zeros(N, dtype=jnp.float32).at[H:H + S].set(d_sector)
    x0 = jnp.linalg.solve(jnp.eye(N) - A, d0)

    # owners: dormant until their arrival tick
    arrival = jnp.full(N, 10 ** 9, dtype=jnp.int32)
    arrival = arrival.at[H + S:].set(
        cfg.first_arrival + cfg.arrival_spacing * jnp.arange(O, dtype=jnp.int32))
    home = jnp.zeros(N, dtype=jnp.int32)
    home = home.at[H + S:].set(H + (jnp.arange(O, dtype=jnp.int32) % S))

    node_types = jnp.concatenate([
        jnp.zeros(H, dtype=jnp.int32),
        jnp.ones(S, dtype=jnp.int32),
        jnp.full((O,), 2, dtype=jnp.int32),
    ])
    is_household = (node_types == 0).astype(jnp.float32)
    node_attrs = {
        "gross_output": x0.astype(jnp.float32),
        "spend_pref": spend_pref,
        "spend_weights": jnp.zeros((N, S), dtype=jnp.float32),
        "demand_h": d0,
        "demand_k": jnp.zeros(N, dtype=jnp.float32),
        "last_reward": income0,
        "capital_income": jnp.zeros(N, dtype=jnp.float32),
        "capital": jnp.zeros(N, dtype=jnp.float32),          # owner slots: K_i
        "pub_cap": jnp.zeros(N, dtype=jnp.float32),          # sector slots: K^pub_j
        "pub_profit": jnp.zeros(N, dtype=jnp.float32),       # sector slots: [π^pub]₊
        "wealth": (cfg.sigma_s / cfg.sigma_d) * cfg.init_income * is_household,
        "active": is_household + (node_types == 1).astype(jnp.float32),
        "arrival_tick": arrival,
        "home_sector": home,
    }
    global_attrs = {
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
        "upkeep_paid": jnp.array(0.0, dtype=jnp.float32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"technical": A},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: CapitalEconomyConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
