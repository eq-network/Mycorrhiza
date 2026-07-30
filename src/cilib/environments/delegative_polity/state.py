"""
Initial-state factory for Delegative Polity.

The delegation matrix starts as a row-normalized Erdos-Renyi digraph (everyone
spreads their delegated share over a random subset) plus a fixed
``self_weight`` diagonal — the vote you always keep. Heterogeneous out-degrees
make the left eigenvector heterogeneous from t=0, the seed preferential
attachment amplifies. AI rows delegate uniformly and are frozen by dynamics:
their power comes from being delegated TO, not from where their voice goes
(the influence_exchange reservoir idiom — an absorbing AI row would drain all
voice to the reservoir by construction and the readout could not discriminate).

``ideal`` and ``endowment`` are drawn once and never rewritten: preferences
are exogenous ideal points (Black/Downs spatial voting), income heterogeneity
is deliberately thin (this is not a rebuild of capital_economy). Globals any
transform writes are jnp arrays from t=0 (pytree-treedef stability under
``lax.scan``).
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import erdos_renyi
from .config import DelegativePolityConfig


# --- resources and type contracts (docs/ledger-design.md §2; docs/gd-suite-v0.1.md §2) --
#
# Three kinds of field exist here, declared so the coupled rewrite and the
# suite tooling can consume them instead of re-deriving them from the code.

# Conserved ledgers — stocks that only ever reallocate (or mint through the
# named sources). The ladder's ledger rung asserts these contracts hold.
LEDGERS = {
    "delegation": {
        "kind": "adjacency",             # per-ROW conservation: each row sums to 1
        "units": "ballot share",
        "sources": (),                   # pure reallocation — nothing mints voice
        "sinks": (),
        "moved_by": ("rewire_delegation", "sortition"),
        "floor": "self_weight (erodible only via franchise_erosion x regime)",
    },
    "wealth": {
        "kind": "node stock",
        "units": "money",
        "sources": ("tax_and_redistribute",),   # mints sum(endowment) per tick
        "sinks": (),                     # write-only score: never spent (the
                                         # WP1-unification seam, gd-suite §4.1)
        "moved_by": ("tax_and_redistribute",),  # flat tax + equal payout, conserving
    },
}

# Ports — computed collective variables and rate-layer fields (never stocks).
# Cross-domain couplings may read ONLY ledgers and ports (ledger-design §5).
PORTS = {
    "influence": "power: normalized ballots held, re-issued each tick (tally_power)",
    "policy_target": "the enacted tax rate (power_weighted_vote)",
    "enforcement": "rule in practice, regime-gated (update_regime)",
    "redelegation_friction": "freedom to re-delegate, regime-gated (update_regime)",
    "amplification": "the scheduled threat: clock-written attractiveness multiplier",
    "cap_scale": "defense seam (influence_cap writes it)",
    "attract_boost": "coupling seam (a persuasion domain would write it)",
    "engagement": "the open action channel (BroadcastPolicy closes it)",
}


def make_delegation(cfg: DelegativePolityConfig, key) -> jnp.ndarray:
    """Row-stochastic D: self_weight on the diagonal, the rest spread uniformly
    over an Erdos-Renyi neighbor draw (guarded so no row is empty)."""
    N = cfg.n_citizens + cfg.n_ai
    A = erdos_renyi(N, cfg.p_delegate, key)
    A = A * (1.0 - jnp.eye(N))                      # no self-edge in the draw
    deg = jnp.sum(A, axis=1, keepdims=True)
    # an isolated row falls back to delegating uniformly to everyone else
    uniform = (jnp.ones((N, N)) - jnp.eye(N)) / (N - 1)
    offdiag = jnp.where(deg > 0, A / jnp.maximum(deg, 1.0), uniform)
    return cfg.self_weight * jnp.eye(N) + (1.0 - cfg.self_weight) * offdiag


def make_state(cfg: DelegativePolityConfig, key) -> GraphState:
    N = cfg.n_citizens + cfg.n_ai
    k_net, k_pref, k_endow, k_run = jr.split(key, 4)

    node_types = (jnp.arange(N) >= cfg.n_citizens).astype(jnp.int32)
    is_ai = node_types == 1

    ideal = jnp.where(
        is_ai, cfg.ai_bias,
        jnp.clip(cfg.true_rate + cfg.pref_noise * jr.normal(k_pref, (N,)), 0.0, 1.0))
    endowment = jnp.where(
        is_ai, 0.0,
        jr.uniform(k_endow, (N,), minval=cfg.endow_low, maxval=cfg.endow_high))

    node_attrs = {
        "ideal": ideal.astype(jnp.float32),
        "endowment": endowment.astype(jnp.float32),
        "wealth": jnp.zeros(N, dtype=jnp.float32),
        "position": ideal.astype(jnp.float32),
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
        "policy_target": jnp.array(cfg.true_rate, dtype=jnp.float32),
        "enforcement": jnp.array(1.0, dtype=jnp.float32),
        "redelegation_friction": jnp.array(1.0, dtype=jnp.float32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={"delegation": make_delegation(cfg, k_net)},
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: DelegativePolityConfig):
    """``key -> GraphState`` (independent per-seed init — each seed draws its
    own delegation graph, ideals, and endowments)."""
    return lambda key: make_state(cfg, key)
