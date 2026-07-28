"""`make_reducers` must score what `make_metrics` scores — at O(1) memory.

The register's deliverable is a measured number per (S, P) corner. A cheaper
path to that number is only worth having if it is the SAME number, so this
pins the streaming form against the trace form at every corner and on the
batched sweep path the benchmark harness uses.

`allclose`, not bit-equality: `late_mean` accumulates in step order while
`jnp.mean` over a materialized (T, N) block reduces as a tree. Unlike the
sparse-adjacency swap — which was exact — fusing a fold changes summation
order by construction.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.value_contagion import make_reducers

CORNERS = {
    "pluralism": dict(ai_homophily=0.05, p_advantage=1.0),
    "assimilation": dict(ai_homophily=0.05, p_advantage=6.0),
    "parallel_cultures": dict(ai_homophily=0.9, p_advantage=1.0),
    "displacement": dict(ai_homophily=0.9, p_advantage=6.0),
}


def test_reducer_matches_the_metric_at_every_corner():
    for name, overrides in CORNERS.items():
        env = make_env("value_contagion", **overrides)
        reducers = make_reducers(env.config)

        _, trace = env.run(jr.PRNGKey(4), n_steps=150)
        from_trace = env.metrics["human_origin_share"](trace)

        _, _, reduced = env.run_reduced(jr.PRNGKey(4), 150, reducers)
        assert jnp.allclose(reduced["human_origin_share"], from_trace, rtol=1e-5), \
            f"{name}: {reduced['human_origin_share']} vs {from_trace}"


def test_reducer_matches_across_a_batched_sweep():
    """The sweep path: `reduced[name]` must be the same per-seed vector the
    harness would have bootstrapped from the full trajectories."""
    env = make_env("value_contagion", ai_homophily=0.9, p_advantage=6.0)
    reducers = make_reducers(env.config)

    _, trace = env.run_batch(jr.PRNGKey(7), n_seeds=12, n_steps=150)
    from_trace = jax.vmap(env.metrics["human_origin_share"])(trace)

    _, _, reduced = env.run_reduced_batch(jr.PRNGKey(7), 12, 150, reducers)
    assert reduced["human_origin_share"].shape == (12,)
    assert jnp.allclose(reduced["human_origin_share"], from_trace, rtol=1e-5)


def test_reduced_path_keeps_no_trajectory_by_default():
    """The point of the exercise: nothing (T, N)-shaped is retained."""
    env = make_env("value_contagion")
    _, trace, reduced = env.run_reduced(jr.PRNGKey(0), 100,
                                        make_reducers(env.config))
    assert trace is None
    assert reduced["human_origin_share"].shape == ()


def test_cheap_scalar_series_still_available():
    """Reducing the AGENT axis is what saves memory; a (T,) scalar series is
    ~8 KB at T=2000, so timelines survive the switch."""
    env = make_env("value_contagion")
    n_h = env.config.n_agents - env.config.n_ai
    _, trace, reduced = env.run_reduced(
        jr.PRNGKey(0), 100, make_reducers(env.config),
        trace_fn=lambda s: {"share": jnp.mean(1.0 - s.node_attrs["culture"][:n_h])})

    assert trace["share"].shape == (100,)                    # (T,), not (T, N)
    # the reducer is the late-window mean of exactly that series
    assert jnp.allclose(reduced["human_origin_share"],
                        jnp.mean(trace["share"][3 * 100 // 4:]), rtol=1e-5)


def test_works_with_the_sparse_representation():
    """Both memory fixes compose: sparse state AND streaming metrics."""
    env = make_env("value_contagion", sparse_friendship=True, p_advantage=6.0)
    dense_env = make_env("value_contagion", p_advantage=6.0)
    r = make_reducers(env.config)
    _, _, sparse_red = env.run_reduced(jr.PRNGKey(2), 120, r)
    _, _, dense_red = dense_env.run_reduced(jr.PRNGKey(2), 120, r)
    assert jnp.allclose(sparse_red["human_origin_share"],
                        dense_red["human_origin_share"], rtol=1e-6)
