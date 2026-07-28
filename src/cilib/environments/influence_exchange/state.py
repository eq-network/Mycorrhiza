"""
Initial-state factory for Influence Exchange.

The listening matrix starts as a row-normalized Erdos-Renyi digraph (everyone
listens uniformly to a random subset) plus a fixed ``self_weight`` diagonal.
Heterogeneous out-degrees make the left eigenvector heterogeneous from t=0 —
the seed that preferential attachment amplifies. AI rows listen too (uniformly,
frozen by dynamics): their power comes from being listened TO, not from what
they hear.

``influence`` starts uniform (1/N) and is tracked in-loop by power iteration —
one v <- normalize(W^T v) per tick — so for a frozen W it converges to the
DeGroot consensus weights (Golub-Jackson), and for an evolving W it tracks the
instantaneous eigenvector. ``cap_scale`` and ``attract_boost`` start at 1.0:
the former is the influence_cap mechanism's write target, the latter is the
A5 coupling seam (persuasion_shifts_politics writes it; standalone runs never
touch it). ``last_reward`` exists for the ``GameSpec.rewards`` contract (no
payoff rule in v0). Globals any transform writes are jnp arrays from t=0
(pytree-treedef stability under ``lax.scan``).
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import erdos_renyi
from .config import InfluenceExchangeConfig


def make_listening(cfg: InfluenceExchangeConfig, key) -> jnp.ndarray:
    """Row-stochastic W: self_weight on the diagonal, the rest spread uniformly
    over an Erdos-Renyi neighbor draw (guarded so no row is empty)."""
    N = cfg.n_citizens + cfg.n_ai
    A = erdos_renyi(N, cfg.p_listen, key)
    A = A * (1.0 - jnp.eye(N))                      # no self-edge in the draw
    deg = jnp.sum(A, axis=1, keepdims=True)
    # an isolated row falls back to listening uniformly to everyone else
    uniform = (jnp.ones((N, N)) - jnp.eye(N)) / (N - 1)
    offdiag = jnp.where(deg > 0, A / jnp.maximum(deg, 1.0), uniform)
    return cfg.self_weight * jnp.eye(N) + (1.0 - cfg.self_weight) * offdiag


def make_state(cfg: InfluenceExchangeConfig, key) -> GraphState:
    N = cfg.n_citizens + cfg.n_ai
    k_net, k_op, k_run = jr.split(key, 3)

    node_types = (jnp.arange(N) >= cfg.n_citizens).astype(jnp.int32)
    is_ai = node_types == 1

    opinion = jnp.where(
        is_ai, cfg.ai_bias,
        cfg.signal_noise * jr.normal(k_op, (N,)))    # citizens: truth 0 + noise

    node_attrs = {
        "opinion": opinion.astype(jnp.float32),
        "signal": opinion.astype(jnp.float32),   # the anchor: initial, never rewritten
        "influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        "engagement": jnp.ones(N, dtype=jnp.float32),
        "amplification": jnp.ones(N, dtype=jnp.float32),
        "cap_scale": jnp.ones(N, dtype=jnp.float32),
        "attract_boost": jnp.ones(N, dtype=jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),
    }
    global_attrs = {
        "rng_key": k_run,
        "step": jnp.array(0, dtype=jnp.int32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"listening": make_listening(cfg, k_net)},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: InfluenceExchangeConfig):
    """``key -> GraphState`` (independent per-seed init — each seed draws its
    own listening graph and opinion noise)."""
    return lambda key: make_state(cfg, key)
