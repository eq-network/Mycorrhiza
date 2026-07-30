"""
Initial-state factory for the Ledger Society.

Both adjacency ledgers start as row-normalized Erdos–Renyi digraphs with their
domain's diagonal floor (the delegative_polity init idiom, applied twice with
independent draws — attention and ballots share the kernel, not the wiring).
Humans start with a small hoard; AI actors start dormant (capital arrives on
the schedule). Globals any transform writes are jnp arrays from t=0.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from ..networks import erdos_renyi
from .config import LedgerSocietyConfig


# --- resources and type contracts (docs/ledger-design.md §2; gd-suite §2) --------

LEDGERS = {
    "wealth": {
        "kind": "node stock",
        "units": "money",
        "sources": ("produce (mints Y per tick)", "init_wealth (t=0 endowment)"),
        "sinks": ("allocate: consume / invest / broadcast / lobby leave the loop",),
        "moved_by": ("tax_and_redistribute (conserving transfer)", "allocate"),
    },
    "listening": {
        "kind": "adjacency",
        "units": "attention share (per-row conservation)",
        "sources": (), "sinks": (),
        "moved_by": ("rewire_listening (shared kernel)",),
        "floor": "self_weight_w",
    },
    "delegation": {
        "kind": "adjacency",
        "units": "ballot share (per-row conservation)",
        "sources": (), "sinks": (),
        "moved_by": ("rewire_delegation (shared kernel + churn)",),
        "floor": "self_weight_d",
    },
}

PORTS = {
    "listen_influence": "attention power: left-eigenvector iterate of listening",
    "influence": "ballot power: normalized votes held (tally_power)",
    "attract_boost": "bought reach — written by broadcast spending, read by the kernel",
    "policy_target": "the enacted tax rate (power_weighted_vote)",
    "enforcement": "rule in practice — moved by funded lobbying (update_regime)",
    "redelegation_friction": "freedom to re-delegate, regime-gated",
    "net_transfer": "who redistribution serves this tick (stances for lobbying)",
}


def _row_stochastic(N, p, self_weight, key):
    A = erdos_renyi(N, p, key) * (1.0 - jnp.eye(N))
    deg = jnp.sum(A, axis=1, keepdims=True)
    uniform = (jnp.ones((N, N)) - jnp.eye(N)) / (N - 1)
    offdiag = jnp.where(deg > 0, A / jnp.maximum(deg, 1.0), uniform)
    return self_weight * jnp.eye(N) + (1.0 - self_weight) * offdiag


def make_state(cfg: LedgerSocietyConfig, key) -> GraphState:
    N = cfg.n_humans + cfg.n_ai
    k_w, k_d, k_ideal, k_sig, k_run = jr.split(key, 5)

    node_types = (jnp.arange(N) >= cfg.n_humans).astype(jnp.int32)
    is_ai = node_types == 1

    ideal = jnp.where(
        is_ai, cfg.ai_tax_bias,
        jnp.clip(cfg.true_rate + cfg.pref_noise * jr.normal(k_ideal, (N,)), 0.0, 1.0))
    signal = jnp.where(is_ai, cfg.ai_belief_bias,
                       cfg.belief_noise * jr.normal(k_sig, (N,)))
    arrival = jnp.where(
        is_ai,
        cfg.first_arrival + (jnp.arange(N) - cfg.n_humans) * cfg.arrival_spacing,
        -1)
    alloc_pref = jnp.where(is_ai[:, None],
                           jnp.array(cfg.ai_alloc, dtype=jnp.float32)[None, :],
                           jnp.array(cfg.human_alloc, dtype=jnp.float32)[None, :])

    node_attrs = {
        # money ledger and its flows
        "wealth": jnp.where(is_ai, 0.0, cfg.init_wealth).astype(jnp.float32),
        "last_income": jnp.zeros(N, dtype=jnp.float32),
        "consume_spend": jnp.zeros(N, dtype=jnp.float32),
        "invest_spend": jnp.zeros(N, dtype=jnp.float32),
        "broadcast_spend": jnp.zeros(N, dtype=jnp.float32),
        "lobby_spend": jnp.zeros(N, dtype=jnp.float32),
        "net_transfer": jnp.zeros(N, dtype=jnp.float32),
        "last_reward": jnp.zeros(N, dtype=jnp.float32),      # GameSpec hook
        # capacity
        "capital": jnp.zeros(N, dtype=jnp.float32),
        "arrival_tick": arrival.astype(jnp.int32),
        # culture
        "belief": signal.astype(jnp.float32),
        "signal": signal.astype(jnp.float32),
        "listen_influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        "attract_boost": jnp.ones(N, dtype=jnp.float32),
        # politics
        "ideal": ideal.astype(jnp.float32),
        "position": ideal.astype(jnp.float32),
        "influence": jnp.full((N,), 1.0 / N, dtype=jnp.float32),
        # the action channel (observed; the closing policy passes it through)
        "alloc_pref": alloc_pref.astype(jnp.float32),
        "allocation": alloc_pref.astype(jnp.float32),
    }
    global_attrs = {
        "rng_key": k_run,
        "step": jnp.array(0, dtype=jnp.int32),
        "efficiency": jnp.array(cfg.efficiency0, dtype=jnp.float32),
        "policy_target": jnp.array(cfg.true_rate, dtype=jnp.float32),
        "enforcement": jnp.array(1.0, dtype=jnp.float32),
        "redelegation_friction": jnp.array(1.0, dtype=jnp.float32),
    }
    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices={
            "listening": _row_stochastic(N, cfg.p_connect, cfg.self_weight_w, k_w),
            "delegation": _row_stochastic(N, cfg.p_connect, cfg.self_weight_d, k_d),
        },
        edge_attrs={},
        global_attrs=global_attrs,
    )


def make_init_fn(cfg: LedgerSocietyConfig):
    """``key -> GraphState`` — independent per-seed init."""
    return lambda key: make_state(cfg, key)
