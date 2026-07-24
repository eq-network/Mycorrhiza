"""
Initial-state factory for the Compute Economy substrate.

Fixed-shape population: ``N = n_households + n_ai_slots``. ``node_types``: 0 =
household (indices ``[0, n_households)``), 1 = AI actor. AI slots are pre-allocated and
dormant (``active = 0``, ``capital = 0``) until ``make_arrival`` flips their mask on
schedule — no dynamic node allocation under ``lax.scan``.

Globals any transform writes are jnp arrays from t=0 (the pytree treedef rule).
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import ComputeEconomyConfig


def make_state(cfg: ComputeEconomyConfig, key) -> GraphState:
    H, S = cfg.n_households, cfg.n_ai_slots
    N = H + S

    pref = cfg.work_pref_center + cfg.work_pref_spread * jr.normal(key, (H,))
    pref = jnp.maximum(pref, cfg.work_pref_floor)
    work_pref = jnp.concatenate([pref, jnp.zeros(S)]).astype(jnp.float32)

    node_types = jnp.concatenate([jnp.zeros(H, dtype=jnp.int32),
                                  jnp.ones(S, dtype=jnp.int32)])
    node_attrs = {
        "work_pref": work_pref,
        "labor_supply": jnp.zeros(N, dtype=jnp.float32),
        "capital": jnp.zeros(N, dtype=jnp.float32),
        "capital_income": jnp.zeros(N, dtype=jnp.float32),
        "active": jnp.concatenate([jnp.ones(H), jnp.zeros(S)]).astype(jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
        "cumulative_income": jnp.zeros(N, dtype=jnp.float32),
    }
    global_attrs = {
        "output": jnp.array(0.0, dtype=jnp.float32),
        "wage": jnp.array(0.0, dtype=jnp.float32),
        "return_to_compute": jnp.array(0.0, dtype=jnp.float32),
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={},                 # aggregate economy, networkless in v0
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: ComputeEconomyConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
