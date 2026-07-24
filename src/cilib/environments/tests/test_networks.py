"""Behavioral tests for the network-structure generators."""
import jax.numpy as jnp
import jax.random as jr

from cilib.environments.networks import (
    GENERATORS, complete_graph, erdos_renyi, ring_graph, watts_strogatz,
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
