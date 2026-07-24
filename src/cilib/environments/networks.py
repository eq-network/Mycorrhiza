"""
Network-structure generators — shared helpers for graph-structured ABMs.

Not a catalog (no type function, no REGISTRY): plain array generators consumed by
environment builders' ``state.py`` files, the same non-registry-helper role
``commons_metrics.py`` plays for metrics. The epidemic (A3) and influence-exchange (A4)
ABMs draw their trust/influence topologies from here; varying the generator is one of
the suite's "different base assumptions" dials.

All return dense ``(N, N)`` float32 adjacency matrices, symmetric, zero diagonal,
entries in {0, 1} — the existing ``adj_matrices`` convention. Stochastic generators
consume a PRNG key and are jit/vmap-safe (fixed shapes, no Python branching on traced
values).

``watts_strogatz`` uses a **vectorized rewiring approximation**: each ring edge is
independently Bernoulli(p)-gated and, if rewired, its endpoint resamples uniformly from
all other nodes — duplicate-edge collisions are tolerated (the OR with the existing
edge simply keeps it), unlike the sequential original (Watts & Strogatz 1998). Degree
is therefore approximately, not exactly, preserved. Documented trade-off for vmap
safety.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr


def _symmetrize(adj: jnp.ndarray) -> jnp.ndarray:
    """Undirected + no self-loops + {0,1} entries."""
    sym = jnp.maximum(adj, adj.T)
    return sym * (1.0 - jnp.eye(sym.shape[0], dtype=sym.dtype))


def complete_graph(n: int) -> jnp.ndarray:
    """Everyone adjacent to everyone (the well-mixed baseline)."""
    return _symmetrize(jnp.ones((n, n), dtype=jnp.float32))


def ring_graph(n: int, k: int = 2) -> jnp.ndarray:
    """Ring lattice: each node adjacent to its ``k`` nearest neighbours per side."""
    idx = jnp.arange(n)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    circ = jnp.minimum(dist, n - dist)                    # circular distance
    return _symmetrize(((circ >= 1) & (circ <= k)).astype(jnp.float32))


def erdos_renyi(n: int, p: float, key) -> jnp.ndarray:
    """G(n, p): each undirected edge present independently with probability ``p``."""
    draw = jr.uniform(key, (n, n))
    upper = jnp.triu(draw < p, k=1).astype(jnp.float32)   # decide each pair once
    return _symmetrize(upper)


def watts_strogatz(n: int, k: int, p: float, key) -> jnp.ndarray:
    """Small-world: ring lattice with vectorized Bernoulli(p) rewiring (see module
    docstring for the approximation). ``p=0`` reduces exactly to ``ring_graph(n, k)``."""
    k_gate, k_target = jr.split(key)
    ring = jnp.triu(ring_graph(n, k), k=1)                # each ring edge once
    rewire = ring * jr.bernoulli(k_gate, p=p, shape=(n, n)).astype(jnp.float32)
    kept = ring - rewire

    # each rewired edge (i, j) becomes (i, t) with t uniform over nodes != i
    targets = jr.randint(k_target, (n, n), 0, n - 1)
    row = jnp.arange(n)[:, None]
    targets = targets + (targets >= row)                  # skip self, stay in [0, n)
    new_edges = jnp.zeros((n, n), dtype=jnp.float32)
    new_edges = new_edges.at[row * jnp.ones((n, n), dtype=jnp.int32), targets].max(rewire)
    return _symmetrize(jnp.maximum(kept, new_edges))


# name -> generator (plain dict for discoverability; not a formal catalog).
GENERATORS = {
    "complete": complete_graph,
    "ring": ring_graph,
    "erdos_renyi": erdos_renyi,
    "watts_strogatz": watts_strogatz,
}
