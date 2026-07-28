"""``sparse_friendship`` is a representation swap, not a model change.

The CLAUDE.md rule for re-composing a paradigm — assert the new path is
numerically identical to the old one for a fixed seed before trusting it —
applied to a change of storage type. If any assertion here loosens, the sparse
path has stopped being the same model and the benchmark's speedup is measuring
a different simulation.

A note on exactness. Dense and sparse matvec reduce in different orders, so they
agree only to float32 rounding *in general* (~1e-7 relative at N >~ 200 on
arbitrary vectors). The trajectories below are nevertheless bit-identical, and
that is not luck: the vectors ``adopt`` contracts against are ``log_keep * c``
with binary ``c``, so both paths sum the same few distinct magnitudes. The
substrate then thresholds these through ``jr.bernoulli``, which would amplify
even a 1-ulp disagreement into a discrete flip — so bit-equality is the honest
assertion to make here, and a loosened tolerance would hide exactly the failure
worth catching.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import sparse

from cilib.agents.broadcast import BroadcastPolicy
from cilib.environments import make_env
from cilib.environments.game import close
from cilib.environments.value_contagion import (
    ValueContagionConfig, build_game, make_state, observe_fn,
)

CORNERS = {                       # the (S, P) corners from cultural-register §11
    "pluralism": dict(ai_homophily=0.05, p_advantage=1.0),
    "assimilation": dict(ai_homophily=0.05, p_advantage=6.0),
    "parallel_cultures": dict(ai_homophily=0.9, p_advantage=1.0),
    "displacement": dict(ai_homophily=0.9, p_advantage=6.0),
}


def test_state_holds_the_same_graph_in_both_representations():
    dense = make_state(ValueContagionConfig(), jr.PRNGKey(0))
    spars = make_state(ValueContagionConfig(sparse_friendship=True), jr.PRNGKey(0))

    W_d, W_s = dense.adj_matrices["friendship"], spars.adj_matrices["friendship"]
    assert isinstance(W_s, sparse.BCOO) and not isinstance(W_d, sparse.BCOO)
    assert bool(jnp.all(W_s.todense() == W_d))
    # everything else about the state is untouched by the swap
    assert bool(jnp.all(dense.node_types == spars.node_types))
    assert bool(jnp.all(dense.node_attrs["culture"] == spars.node_attrs["culture"]))
    assert bool(jnp.all(dense.global_attrs["rng_key"] == spars.global_attrs["rng_key"]))


def test_observation_is_identical():
    """The boundary a policy sees must not depend on how the graph is stored —
    this is the path that used the jnp.sum(W, axis=1) form."""
    for name, overrides in CORNERS.items():
        obs_d = observe_fn(make_state(ValueContagionConfig(**overrides), jr.PRNGKey(1)))
        obs_s = observe_fn(make_state(
            ValueContagionConfig(sparse_friendship=True, **overrides), jr.PRNGKey(1)))
        assert bool(jnp.all(obs_d == obs_s)), name


def test_trajectory_is_identical_at_every_corner():
    """The load-bearing check: same seed, same config, both representations ->
    the same culture trajectory tick for tick, at all four regimes."""
    for name, overrides in CORNERS.items():
        _, td = make_env("value_contagion", **overrides).run(jr.PRNGKey(4), n_steps=120)
        _, ts = make_env("value_contagion", sparse_friendship=True,
                         **overrides).run(jr.PRNGKey(4), n_steps=120)
        assert bool(jnp.all(td["culture"] == ts["culture"])), name


def test_batched_trajectories_and_metrics_are_identical():
    """vmap over seeds is the tier that forces a static nse; verify the bound
    does not quietly truncate any seed's draw (a truncated graph would spread
    culture differently, so the trajectory equality catches it)."""
    kw = dict(ai_homophily=0.9, p_advantage=6.0)
    env_d = make_env("value_contagion", **kw)
    env_s = make_env("value_contagion", sparse_friendship=True, **kw)

    fd, trace_d = env_d.run_batch(jr.PRNGKey(7), n_seeds=12, n_steps=150)
    fs, trace_s = env_s.run_batch(jr.PRNGKey(7), n_seeds=12, n_steps=150)

    assert bool(jnp.all(trace_d["culture"] == trace_s["culture"]))
    assert bool(jnp.all(fs.adj_matrices["friendship"].todense()
                        == fd.adj_matrices["friendship"]))
    share = lambda env, tr: jax.vmap(env.metrics["human_origin_share"])(tr)
    assert bool(jnp.all(share(env_d, trace_d) == share(env_s, trace_s)))


def test_open_game_boundary_is_unchanged():
    """The swap must survive the GameSpec path too, not just the closed env."""
    kw = dict(p_advantage=6.0, recovery=0.0)
    policy = BroadcastPolicy(effort=0.5)      # non-uniform effort exercises beta_src
    _, td = close(build_game(**kw), policy).run(jr.PRNGKey(5), n_steps=60)
    _, ts = close(build_game(sparse_friendship=True, **kw),
                  policy).run(jr.PRNGKey(5), n_steps=60)
    assert bool(jnp.all(td["culture"] == ts["culture"]))


def test_sparse_state_is_jit_safe():
    env = make_env("value_contagion", sparse_friendship=True)
    state = env.init_fn(jr.PRNGKey(3))
    new_state = jax.jit(env.round_fn)(state, 0, jr.PRNGKey(4))
    assert int(new_state.global_attrs["step"]) == 1
    assert isinstance(new_state.adj_matrices["friendship"], sparse.BCOO)
    c = new_state.node_attrs["culture"]
    assert bool(jnp.all((c == 0.0) | (c == 1.0)))
