"""Behavioral tests for the sparse adjacency representation (networks.py).

The claim under test is narrow and load-bearing: converting a generator's dense
draw to BCOO is *lossless* and *cheaper*, never a different graph. Three risks
get pinned here — silent truncation, a bound that is too tight for the configs
the register actually runs, and the two ``sum``-shaped API traps.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.experimental import sparse

from cilib.environments.networks import (
    erdos_renyi, row_sums, sparse_nse_bound, to_sparse, typed_homophily,
)


def _dense(n=60, n_ai=12, hom=0.05, key=0):
    return typed_homophily(n, n_ai, 6.0, hom, jr.PRNGKey(key))


def test_roundtrip_is_lossless():
    """The representation swap preserves the graph exactly — bit-for-bit, not
    approximately. Everything else here depends on this."""
    d = _dense()
    W = to_sparse(d, sparse_nse_bound(60, 6.0))
    assert isinstance(W, sparse.BCOO)
    assert W.shape == d.shape
    assert bool(jnp.all(W.todense() == d))


def test_padding_above_actual_count_is_harmless():
    """nse must be static for vmap, so it is a BOUND, not the true count. Padding
    to well above the draw must not perturb the matrix or the matvec."""
    d = _dense()
    actual = int(jnp.count_nonzero(d))
    x = jr.uniform(jr.PRNGKey(1), (d.shape[0],))
    for pad in (0, 1, 500):
        W = to_sparse(d, actual + pad)
        assert bool(jnp.all(W.todense() == d))
        assert bool(jnp.allclose(W @ x, d @ x, atol=1e-5))


def test_under_provisioned_nse_raises_instead_of_truncating():
    """BCOO.fromdense drops trailing nonzeros silently — a truncated graph is a
    valid-looking sparser one. to_sparse must refuse rather than corrupt."""
    d = _dense()
    actual = int(jnp.count_nonzero(d))
    with pytest.raises(ValueError, match="truncate"):
        to_sparse(d, actual - 1)


def test_nse_bound_holds_across_the_register_corners():
    """The bound is the only thing standing between a vmapped run and silent
    truncation (the eager guard cannot fire under trace). Check it against real
    draws at both separations, over many seeds."""
    for n, n_ai in ((40, 8), (200, 40), (600, 120)):
        bound = sparse_nse_bound(n, 6.0)
        assert bound < n * n              # otherwise sparsity buys nothing
        for hom in (0.05, 0.9):           # axis S endpoints the corners use
            worst = max(int(jnp.count_nonzero(typed_homophily(n, n_ai, 6.0, hom,
                                                              jr.PRNGKey(s))))
                        for s in range(25))
            assert worst <= bound, f"n={n} hom={hom}: {worst} > bound {bound}"


def test_row_sums_matches_dense_and_returns_dense():
    """jnp.sum raises on BCOO and W.sum(axis=1) returns another BCOO; row_sums
    hides both. Degrees feed jnp.maximum, so the result must be dense."""
    d = _dense()
    W = to_sparse(d, sparse_nse_bound(60, 6.0))
    expected = jnp.sum(d, axis=1)

    assert not isinstance(row_sums(W), sparse.BCOO)
    assert bool(jnp.all(row_sums(W) == expected))
    assert bool(jnp.all(row_sums(d) == expected))     # dense path unchanged

    with pytest.raises(TypeError):                    # the trap being papered over
        jnp.sum(W, axis=1)


def test_sparse_survives_jit_vmap_and_scan():
    """BCOO is itself a pytree, so it composes inside GraphState's registered
    pytree with no change to core/graph.py — the whole plan rests on this."""
    n, nse = 40, sparse_nse_bound(40, 6.0)

    def rollout(key):
        W = to_sparse(erdos_renyi(n, 6.0 / (n - 1), key), nse)
        x0 = jr.uniform(key, (n,))
        step = lambda x, _: ((W @ x) / jnp.maximum(row_sums(W), 1.0), x)
        return jax.lax.scan(step, x0, None, length=6)

    final, hist = jax.jit(jax.vmap(rollout))(jr.split(jr.PRNGKey(0), 4))
    assert final.shape == (4, n) and hist.shape == (4, 6, n)
    assert bool(jnp.all(jnp.isfinite(final)))


def test_generators_are_all_convertible():
    """Sparsity is a property of the representation, not of one generator."""
    n = 50
    for d in (erdos_renyi(n, 0.1, jr.PRNGKey(0)),
              typed_homophily(n, 10, 6.0, 0.5, jr.PRNGKey(0))):
        W = to_sparse(d, int(jnp.count_nonzero(d)))
        assert bool(jnp.all(W.todense() == d))
