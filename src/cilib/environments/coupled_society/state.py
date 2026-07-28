"""
Initial-state factory for the Coupled Society: the UNION of the three domains'
states over one shared population (AI-last convention everywhere).

Every field keeps its home substrate's name and init, so the imported
transform factories run unchanged. The only genuinely shared fields are by
design: ``last_reward`` (written by the economy's income distribution — the
other two domains' v0s never write it), ``rng_key``/``step``, and the new
``enforcement`` global (written by politics_rewrites_market_rules, read by the
enforced tax). Two adjacency matrices coexist because ``adj_matrices`` is
already plural — the friendship graph (culture) and the listening matrix
(politics) are different relations over the same nodes.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import typed_homophily
from ..influence_exchange.state import make_listening
from .config import CoupledSocietyConfig


def make_state(cfg: CoupledSocietyConfig, key) -> GraphState:
    H, S = cfg.n_humans, cfg.n_ai
    N = H + S
    k_pref, k_net, k_listen, k_op, k_run = jr.split(key, 5)
    econ, politics = cfg.econ(), cfg.politics()

    node_types = jnp.concatenate([jnp.zeros(H, dtype=jnp.int32),
                                  jnp.ones(S, dtype=jnp.int32)])
    is_ai = node_types == 1

    pref = econ.work_pref_center + econ.work_pref_spread * jr.normal(k_pref, (H,))
    work_pref = jnp.concatenate([jnp.maximum(pref, econ.work_pref_floor),
                                 jnp.zeros(S)]).astype(jnp.float32)

    opinion = jnp.where(is_ai, politics.ai_bias,
                        politics.signal_noise * jr.normal(k_op, (N,)))

    node_attrs = {
        # economy (compute_economy/state.py init, verbatim semantics)
        "work_pref": work_pref,
        "labor_supply": jnp.zeros(N, dtype=jnp.float32),
        "capital": jnp.zeros(N, dtype=jnp.float32),
        "capital_income": jnp.zeros(N, dtype=jnp.float32),
        "active": jnp.concatenate([jnp.ones(H), jnp.zeros(S)]).astype(jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
        "cumulative_income": jnp.zeros(N, dtype=jnp.float32),
        # culture (value_contagion/state.py)
        "culture": node_types.astype(jnp.float32),
        "broadcast_effort": jnp.ones(N, dtype=jnp.float32),
        # politics (influence_exchange/state.py)
        "opinion": opinion.astype(jnp.float32),
        "signal": opinion.astype(jnp.float32),
        "influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        "engagement": jnp.ones(N, dtype=jnp.float32),
        "amplification": jnp.ones(N, dtype=jnp.float32),
        "cap_scale": jnp.ones(N, dtype=jnp.float32),
        "attract_boost": jnp.ones(N, dtype=jnp.float32),
    }
    global_attrs = {
        "output": jnp.array(0.0, dtype=jnp.float32),
        "wage": jnp.array(0.0, dtype=jnp.float32),
        "return_to_compute": jnp.array(0.0, dtype=jnp.float32),
        "enforcement": jnp.array(1.0, dtype=jnp.float32),
        "rng_key": k_run,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={
            "friendship": typed_homophily(N, S, cfg.mean_degree, cfg.ai_homophily, k_net),
            "listening": make_listening(politics, k_listen),
        },
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: CoupledSocietyConfig):
    """``key -> GraphState`` (independent per-seed init)."""
    return lambda key: make_state(cfg, key)
