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

**Sparse representation** (``to_sparse`` / ``sparse_nse_bound`` / ``row_sums``): real
social graphs have mean degree ≪ N, but XLA cannot see that in a dense ``(N, N)``
array — a 99%-zero matrix still costs the full O(N²) FLOPs in ``W @ x``. Converting a
generator's output to ``jax.experimental.sparse.BCOO`` makes the sparsity structural,
so the cost tracks edges rather than node pairs. The generators themselves stay dense:
you draw as before and convert, which keeps the connectivity distribution provably
identical to the dense path (the equivalence test relies on exactly this).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import sparse


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


def typed_homophily(n: int, n_ai: int, mean_degree: float, homophily: float,
                    key) -> jnp.ndarray:
    """Two-type planted partition with degree correction — axis S of the cultural
    register (docs/cultural-register-design.md §2).

    The last ``n_ai`` nodes are type 1 (AI); environments building ``node_types``
    must use the same AI-last convention. Cross-type edges appear with probability
    ``(1 - homophily) * mean_degree / (n - 1)``; within-type probabilities are then
    solved so EVERY node's expected degree stays ``mean_degree`` at every
    ``homophily`` — turning the separation dial must not also move the epidemic
    threshold (degree), or axis S confounds with axis P. Consequence: the small AI
    block grows internally dense as it separates.

    ``homophily=0`` reproduces ``erdos_renyi(n, mean_degree/(n-1), key)`` exactly
    (same key, same draws). ``homophily=1.0`` removes all cross-type edges and
    disconnects the graph (λ₂ = 0, the Fiedler partition is undefined) — studies
    should stay at ≲ 0.9. Within-type probabilities clip at 1 for tiny groups.
    """
    n_h = n - n_ai
    p_mix = mean_degree / (n - 1)
    p_out = jnp.clip((1.0 - homophily) * p_mix, 0.0, 1.0)
    p_in_h = jnp.clip((mean_degree - p_out * n_ai) / max(n_h - 1, 1), 0.0, 1.0)
    p_in_ai = jnp.clip((mean_degree - p_out * n_h) / max(n_ai - 1, 1), 0.0, 1.0)

    is_ai = jnp.arange(n) >= n_h
    same_ai = is_ai[:, None] & is_ai[None, :]
    same_h = (~is_ai)[:, None] & (~is_ai)[None, :]
    P = jnp.where(same_ai, p_in_ai, jnp.where(same_h, p_in_h, p_out))
    upper = jnp.triu(jr.uniform(key, (n, n)) < P, k=1).astype(jnp.float32)
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


# --- sparse representation -------------------------------------------------------

def sparse_nse_bound(n: int, mean_degree: float, sigmas: float = 6.0) -> int:
    """A static upper bound on the stored-entry count of a degree-``mean_degree``
    draw on ``n`` nodes — the ``nse`` to hand ``to_sparse``.

    ``nse`` must be **static** (config-derived, not drawn): it is the leaf shape of
    the BCOO's index/value arrays, and ``vmap`` requires every seed in a batch to
    share it. A stochastic generator's actual edge count varies per key, so the
    bound has to hold for all of them.

    Each undirected edge is stored twice (both ``(i, j)`` and ``(j, i)``), so the
    mean is ``n * mean_degree`` — NOT half of it. Treating the upper triangle as
    ``n(n-1)/2`` independent Bernoulli(``mean_degree/(n-1)``) draws gives
    ``Var[nse] <= 2 * n * mean_degree``, hence the ``sigmas``-deviation headroom.
    At ``n=1000, mean_degree=6`` this is ~6.7k of a possible 1e6 entries.

    Capped at ``n * n``: past that the dense form is stored in full anyway.
    """
    mean = n * mean_degree
    return int(min(math.ceil(mean + sigmas * math.sqrt(2.0 * mean)), n * n))


def to_sparse(dense: jnp.ndarray, nse: int) -> sparse.BCOO:
    """Convert a dense adjacency to BCOO with a fixed ``nse`` (see above).

    **Under-provisioning silently drops edges.** ``BCOO.fromdense`` keeps the first
    ``nse`` nonzeros in row-major order and discards the rest with no error — a
    truncated graph looks like a valid, sparser one. Because that failure is
    invisible downstream, this raises eagerly whenever ``dense`` is concrete
    (single runs, tests, benchmarks). Under ``vmap``/``jit`` the count is a tracer
    and cannot be checked, so the bound from ``sparse_nse_bound`` is doing real
    work there; ``tests/test_sparse_networks.py`` pins that it holds.
    """
    if not isinstance(dense, jax.core.Tracer):
        actual = int(jnp.count_nonzero(dense))
        if actual > nse:
            raise ValueError(
                f"to_sparse would truncate: {actual} nonzeros exceeds nse={nse}. "
                f"Raise the bound (sparse_nse_bound(..., sigmas=...)) or pass nse "
                f"explicitly — silently dropping edges would corrupt the graph.")
    return sparse.BCOO.fromdense(dense, nse=nse)


def row_sums(W) -> jnp.ndarray:
    """Row sums of an adjacency as a DENSE ``(N,)`` array, dense or BCOO input.

    Two traps this papers over: ``jnp.sum(W, axis=1)`` raises ``TypeError`` on a
    BCOO (the free function demands an ndarray), and the method form
    ``W.sum(axis=1)`` returns *another BCOO* rather than a dense vector. Degrees
    feed dense-only ops (``jnp.maximum``, division), so densify here.
    """
    s = W.sum(axis=1)
    return s.todense() if isinstance(s, sparse.BCOO) else s


# name -> generator (plain dict for discoverability; not a formal catalog).
GENERATORS = {
    "complete": complete_graph,
    "ring": ring_graph,
    "erdos_renyi": erdos_renyi,
    "typed_homophily": typed_homophily,
    "watts_strogatz": watts_strogatz,
}
