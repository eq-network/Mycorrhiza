"""Behavioral tests for the network-structure generators."""
import jax.numpy as jnp
import jax.random as jr

from cilib.environments.networks import (
    GENERATORS, complete_graph, erdos_renyi, ring_graph, typed_homophily,
    watts_strogatz,
)


def _check_invariants(adj, n):
    assert adj.shape == (n, n)
    assert adj.dtype == jnp.float32
    assert bool(jnp.all(adj == adj.T)), "symmetric"
    assert float(jnp.sum(jnp.diag(adj))) == 0.0, "zero diagonal"
    assert bool(jnp.all((adj == 0.0) | (adj == 1.0))), "binary entries"


def test_all_generators_satisfy_adjacency_invariants():
    n, key = 12, jr.PRNGKey(0)
    for name, gen in GENERATORS.items():
        if name == "complete":
            adj = gen(n)
        elif name == "ring":
            adj = gen(n, 2)
        elif name == "erdos_renyi":
            adj = gen(n, 0.3, key)
        elif name == "typed_homophily":
            adj = gen(n, 3, 4.0, 0.7, key)
        else:
            adj = gen(n, 2, 0.2, key)
        _check_invariants(adj, n)


def test_complete_and_ring_degrees_are_exact():
    n = 10
    assert bool(jnp.all(jnp.sum(complete_graph(n), axis=1) == n - 1))
    assert bool(jnp.all(jnp.sum(ring_graph(n, k=2), axis=1) == 4))


def test_erdos_renyi_density_tracks_p():
    n, p = 40, 0.25
    densities = []
    for seed in range(8):
        adj = erdos_renyi(n, p, jr.PRNGKey(seed))
        densities.append(float(jnp.sum(adj)) / (n * (n - 1)))
    mean_density = sum(densities) / len(densities)
    assert abs(mean_density - p) < 0.05


def test_watts_strogatz_p_zero_is_the_ring():
    n, k = 14, 2
    ws = watts_strogatz(n, k, 0.0, jr.PRNGKey(3))
    assert bool(jnp.all(ws == ring_graph(n, k)))


def test_stochastic_generators_are_key_reproducible():
    key = jr.PRNGKey(7)
    assert bool(jnp.all(erdos_renyi(20, 0.3, key) == erdos_renyi(20, 0.3, key)))
    assert bool(jnp.all(watts_strogatz(20, 2, 0.4, key) == watts_strogatz(20, 2, 0.4, key)))
    assert bool(jnp.all(typed_homophily(20, 5, 4.0, 0.6, key)
                        == typed_homophily(20, 5, 4.0, 0.6, key)))


# --- typed_homophily: axis S of the cultural register ----------------------------

def test_typed_homophily_zero_is_erdos_renyi():
    n, d, key = 30, 5.0, jr.PRNGKey(4)
    assert bool(jnp.all(typed_homophily(n, 6, d, 0.0, key)
                        == erdos_renyi(n, d / (n - 1), key)))


def test_typed_homophily_degree_correction_holds_across_h():
    """Turning the separation dial must not move expected degree (else axis S
    confounds with the epidemic threshold)."""
    n, n_ai, d = 40, 8, 6.0
    for h in (0.0, 0.5, 0.9):
        degrees = [float(jnp.mean(jnp.sum(
            typed_homophily(n, n_ai, d, h, jr.PRNGKey(s)), axis=1)))
            for s in range(8)]
        mean_degree = sum(degrees) / len(degrees)
        assert abs(mean_degree - d) < 0.6, f"h={h}: {mean_degree}"


def test_typed_homophily_cross_edges_fall_as_h_rises():
    n, n_ai, d = 40, 8, 6.0
    n_h = n - n_ai

    def mean_cross(h):
        return sum(float(jnp.sum(
            typed_homophily(n, n_ai, d, h, jr.PRNGKey(s))[:n_h, n_h:]))
            for s in range(8)) / 8

    crossings = [mean_cross(h) for h in (0.0, 0.3, 0.6, 0.9)]
    assert crossings[0] > crossings[1] > crossings[2] > crossings[3]


def test_homophily_moves_fiedler_alignment_monotonically():
    """Rung (b) of the cultural register: the separation dial and its
    measurement dual move together."""
    from cilib.metrics.families.spectral import fiedler_partition_alignment_of
    n, n_ai, d = 40, 8, 6.0
    types = (jnp.arange(n) >= n - n_ai).astype(jnp.int32)

    def mean_alignment(h):
        return sum(float(fiedler_partition_alignment_of(
            typed_homophily(n, n_ai, d, h, jr.PRNGKey(s)), types))
            for s in range(8)) / 8

    alignments = [mean_alignment(h) for h in (0.0, 0.3, 0.6, 0.9)]
    assert alignments[0] < alignments[1] < alignments[2] < alignments[3]
