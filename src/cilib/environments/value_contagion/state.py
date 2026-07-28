"""
Initial-state factory for Value Contagion.

The friendship network is drawn once per seed by ``networks.typed_homophily``
(axis S) and never rewritten — the evolving array is ``culture``. Being static
is what makes ``cfg.sparse_friendship`` a pure representation swap: nothing
mutates the matrix, so a BCOO never has to grow its stored-entry count. Node types
follow the generator's AI-last convention: indices ``[0, n_agents - n_ai)`` are
human, the rest AI. Culture starts equal to node type: AI agents are born
holding AI-origin culture (the permanent reservoir — dynamics never flip them),
humans holding human-origin. "Spread" is conversion of humans; there is no
patient zero.

``last_reward`` exists only to satisfy the ``GameSpec.rewards`` contract
(no payoff rule in v0; nothing writes it). Globals any transform writes
(``rng_key``, ``step``) are jnp arrays from t=0 — ``GraphState.tree_flatten``
partitions ``global_attrs`` by type, so a Python value that later became an
array would change the pytree treedef mid-``lax.scan``.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import sparse_nse_bound, to_sparse, typed_homophily
from .config import ValueContagionConfig


def make_state(cfg: ValueContagionConfig, key) -> GraphState:
    N = cfg.n_agents
    k_net, k_run = jr.split(key)

    friendship = typed_homophily(N, cfg.n_ai, cfg.mean_degree,
                                 cfg.ai_homophily, k_net)
    if cfg.sparse_friendship:
        # convert the SAME draw rather than generating sparsely: the connectivity
        # distribution is then identical to the dense path by construction
        nse = cfg.sparse_nse or sparse_nse_bound(N, cfg.mean_degree)
        friendship = to_sparse(friendship, nse)
    node_types = (jnp.arange(N) >= N - cfg.n_ai).astype(jnp.int32)

    node_attrs = {
        "culture": node_types.astype(jnp.float32),     # 0 human-origin, 1 AI-origin
        "broadcast_effort": jnp.ones(N, dtype=jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
    }
    global_attrs = {
        "rng_key": k_run,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"friendship": friendship},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: ValueContagionConfig):
    """``key -> GraphState`` for ``run_scan_batch`` (independent per-seed init —
    each seed draws its own network)."""
    return lambda key: make_state(cfg, key)
