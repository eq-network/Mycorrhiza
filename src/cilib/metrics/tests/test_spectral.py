"""Behavioral tests for the spectral metric family (minimal C0 slice)."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments.networks import erdos_renyi
from cilib.metrics.families.spectral import fiedler_partition_alignment_of


def _planted_two_block(n_a=12, n_b=8):
    """Two internally-complete blocks joined by a single bridge edge — the
    Fiedler vector must cut the bridge."""
    n = n_a + n_b
    W = jnp.zeros((n, n), dtype=jnp.float32)
    W = W.at[:n_a, :n_a].set(1.0)
    W = W.at[n_a:, n_a:].set(1.0)
    W = W * (1.0 - jnp.eye(n))
    W = W.at[0, n_a].set(1.0).at[n_a, 0].set(1.0)      # the bridge
    types = (jnp.arange(n) >= n_a).astype(jnp.int32)
    return W, types


def test_recovers_planted_two_block_partition():
    W, types = _planted_two_block()
    assert float(fiedler_partition_alignment_of(W, types)) > 0.9


def test_alignment_is_label_flip_invariant():
    """Eigenvector signs are arbitrary; so is which type is called 1. The score
    must compare partitions, not labelings."""
    W, types = _planted_two_block()
    a = float(fiedler_partition_alignment_of(W, types))
    b = float(fiedler_partition_alignment_of(W, 1 - types))
    assert abs(a - b) < 1e-5


def test_unrelated_partition_scores_near_zero():
    """On an ER graph the Fiedler split has nothing to do with an arbitrary
    8-of-40 type split — the |phi| baseline is ~0, not max-class share."""
    W = erdos_renyi(40, 0.3, jr.PRNGKey(0))
    types = (jnp.arange(40) >= 32).astype(jnp.int32)
    assert float(fiedler_partition_alignment_of(W, types)) < 0.4


def test_alignment_jits_and_vmaps():
    W, types = _planted_two_block()
    jitted = jax.jit(fiedler_partition_alignment_of)
    assert float(jitted(W, types)) > 0.9
    batched = jax.vmap(fiedler_partition_alignment_of, in_axes=(0, None))(
        jnp.stack([W, W]), types)
    assert batched.shape == (2,)


def test_accepts_a_sparse_adjacency():
    """The metric is the measurement dual of value_contagion's separation dial,
    so it must read that env's graph in either representation — same score."""
    from cilib.environments.networks import to_sparse

    W, types = _planted_two_block()
    Ws = to_sparse(W, int(jnp.count_nonzero(W)))
    assert jnp.allclose(fiedler_partition_alignment_of(Ws, types),
                        fiedler_partition_alignment_of(W, types), atol=1e-5)
