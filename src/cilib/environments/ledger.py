"""
Ledger checks — the conservation contract as test helpers (docs/ledger-design.md §2).

A ledger is an ordinary evolving field with a declared conservation contract:
node stocks change only by named sources minus named sinks; adjacency ledgers
are row-stochastic (per-row conservation of a share). These helpers turn the
contract into one-line ladder rungs; they are test utilities, never runtime
checks (the pipeline stays JIT-clean).
"""
from __future__ import annotations

import jax.numpy as jnp


def stock_conservation_error(wealth_before, wealth_after, minted, sunk):
    """|Δ(total stock) − (minted − sunk)| for one tick — 0 under the contract."""
    delta = jnp.sum(wealth_after) - jnp.sum(wealth_before)
    return float(jnp.abs(delta - (jnp.sum(minted) - jnp.sum(sunk))))


def row_stochastic_error(M):
    """max |row sum − 1| — 0 for an adjacency ledger (share conserved per row)."""
    return float(jnp.max(jnp.abs(jnp.sum(M, axis=1) - 1.0)))
