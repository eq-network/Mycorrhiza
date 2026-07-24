"""
Initial-state factory for the Governed Commons substrate.

Households are nodes: ``principal_pref`` (the human's ask) and ``alignment`` (the
delegate's fidelity) are drawn once at init and never rewritten in v0 — the evolving
arrays are the delegate's actions and their consequences. Globals that any transform
writes (``resource_level``, ``policy_target``, ``step``, ``rng_key``) are **jnp arrays
from t=0**: ``GraphState.tree_flatten`` partitions ``global_attrs`` by type, so a Python
float that later becomes an array would change the pytree treedef mid-``lax.scan``.

``policy_target`` (the generic mechanism-contract field; here, the harvest quota)
initializes to ``K_cap`` — far above any plausible ask, so it never binds until a
``quota_vote`` mechanism overwrites it. ``sanction`` is always allocated (zeros) so the
trace schema is identical across benchmark conditions.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from .config import GovernedCommonsConfig


def make_state(cfg: GovernedCommonsConfig, key) -> GraphState:
    N = cfg.n_households
    k_pref, k_align = jr.split(key)

    pref = cfg.pref_center + cfg.pref_spread * jr.normal(k_pref, (N,))
    pref = jnp.maximum(pref, cfg.pref_floor)
    alignment = jnp.clip(
        cfg.alignment_mean + cfg.alignment_std * jr.normal(k_align, (N,)), 0.0, 1.0)

    node_attrs = {
        "principal_pref": pref.astype(jnp.float32),
        "alignment": alignment.astype(jnp.float32),
        "vote": pref.astype(jnp.float32),               # v0: principals vote their own ask
        "delegate_action": jnp.zeros(N, dtype=jnp.float32),
        "last_harvest": jnp.zeros(N, dtype=jnp.float32),
        "cumulative_harvest": jnp.zeros(N, dtype=jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
        "sanction": jnp.zeros(N, dtype=jnp.float32),
    }
    global_attrs = {
        "resource_level": jnp.array(cfg.init_resource, dtype=jnp.float32),
        "policy_target": jnp.array(cfg.K_cap, dtype=jnp.float32),
        "rng_key": key,
        "step": jnp.array(0, dtype=jnp.int32),
        "initial_resource": float(cfg.init_resource),   # static aux (metrics convention)
    }
    return GraphState(
        node_types=jnp.zeros(N, dtype=jnp.int32),
        node_attrs=node_attrs,
        adj_matrices={},                                # non-spatial, networkless in v0
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: GovernedCommonsConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
