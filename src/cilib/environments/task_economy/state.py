"""
Initial-state factory for the Task Economy substrate.

Population: ``N = n_households + 1`` — households (type 0) supply labor; the last node
(type 1) is the AI compute vendor, whose revenue is the rental bill. The task frontier
lives in ``global_attrs`` as an evolving (K,) mask plus the capability scalar — task
state is economy-level, not per-node, and JAX arrays in ``global_attrs`` are dynamic
pytree children, so they trace under ``lax.scan`` like any node attribute.

The economy starts fully manual (no task automated, no compute rented): output
``Y = a_L·L`` and wage ``a_L`` exactly — the pre-frontier steady state every rung
measures against.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import TaskEconomyConfig


def make_state(cfg: TaskEconomyConfig, key) -> GraphState:
    H = cfg.n_households
    N = H + 1                                        # + the compute vendor

    pref = cfg.work_pref_center + cfg.work_pref_spread * jr.normal(key, (H,))
    pref = jnp.maximum(pref, cfg.work_pref_floor)
    work_pref = jnp.concatenate([pref, jnp.zeros(1)]).astype(jnp.float32)

    node_types = jnp.concatenate([jnp.zeros(H, dtype=jnp.int32),
                                  jnp.ones(1, dtype=jnp.int32)])
    node_attrs = {
        "work_pref": work_pref,
        "labor_supply": jnp.zeros(N, dtype=jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
        "cumulative_income": jnp.zeros(N, dtype=jnp.float32),
        "active": jnp.ones(N, dtype=jnp.float32),
    }
    global_attrs = {
        "automated": jnp.zeros(cfg.n_tasks, dtype=jnp.float32),   # the ratchet mask
        "beta_cap": jnp.array(0.0, dtype=jnp.float32),            # capability frontier
        "price_compute": jnp.array(cfg.price_compute0, dtype=jnp.float32),
        "compute_used": jnp.array(0.0, dtype=jnp.float32),
        "output": jnp.array(0.0, dtype=jnp.float32),
        "wage": jnp.array(cfg.a_L, dtype=jnp.float32),            # manual-economy wage
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: TaskEconomyConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
