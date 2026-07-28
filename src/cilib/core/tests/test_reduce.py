"""Behavioral tests for streaming reduction (core/reduce.py + metrics/reducers.py).

The contract: a reducer folded into the scan returns what the equivalent
trace-based metric returns, at O(accumulator) memory instead of O(T x trace).
Every test here is that claim in a different shape — because a reducer that is
merely *cheap* is worthless if it is not also *the same number*.

Tolerance is `allclose`, not bit-equality: a streaming sum accumulates in step
order while `jnp.sum` over a materialized block reduces as a tree, so the two
differ at float32 rounding. That is inherent to fusing the fold, and stated in
metrics/reducers.py rather than hidden behind a loose assertion.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.reduce import Reducer, run_scan_reduce, run_scan_reduce_batch
from cilib.core.scan import run_scan
from cilib.metrics.families.concentration import gini_of
from cilib.metrics.reducers import (
    last, late_mean, per_agent_late_mean, per_agent_total, rate, running_max,
    running_mean, running_min, running_var, total,
)

N = 6


def _init(key):
    """A tiny deterministic-per-key world: each agent's `x` grows by its own rate."""
    return GraphState(
        node_types=jnp.zeros(N, jnp.int32),
        node_attrs={"x": jr.uniform(key, (N,)) + 0.5,
                    "rate": jr.uniform(jr.fold_in(key, 1), (N,)) * 0.1},
        adj_matrices={},
        global_attrs={"step": jnp.array(0, jnp.int32)},
    )


def _round(state, t, key):
    x = state.node_attrs["x"] * (1.0 + state.node_attrs["rate"])
    return state.update_node_attrs("x", x).update_global_attr(
        "step", state.global_attrs["step"] + 1)


def _trace(state):
    return {"x": state.node_attrs["x"]}


X = lambda s: s.node_attrs["x"]
MEAN_X = lambda s: jnp.mean(s.node_attrs["x"])


def _both(reducers, n_steps=40, seed=0):
    """Run once with reducers and once with a full trace; return (reduced, trace)."""
    key = jr.PRNGKey(seed)
    state = _init(key)
    _, _, reduced = run_scan_reduce(_round, state, n_steps, key, reducers)
    _, trace = run_scan(_round, state, n_steps, key, trace_fn=_trace)
    return reduced, trace


def _late(series):                      # the idiom the envs' metrics use
    return series[3 * series.shape[0] // 4:]


# --- each reducer against its trace-based twin -----------------------------------

def test_total_and_running_mean_match_the_trace():
    reduced, trace = _both({"sum": total(MEAN_X), "mean": running_mean(MEAN_X)})
    assert jnp.allclose(reduced["sum"], jnp.sum(jnp.mean(trace["x"], axis=1)), rtol=1e-5)
    assert jnp.allclose(reduced["mean"], jnp.mean(trace["x"]), rtol=1e-5)


def test_late_mean_reproduces_the_late_idiom_exactly():
    """`late_mean(frac=0.25)` must pick the same window as `series[3*T//4:]` —
    including at T not divisible by 4, where an off-by-one would silently shift
    the window and change every register metric's value."""
    for n_steps in (40, 41, 42, 43, 150, 201):
        reduced, trace = _both({"late": late_mean(MEAN_X)}, n_steps=n_steps)
        assert jnp.allclose(reduced["late"], jnp.mean(_late(trace["x"])), rtol=1e-5), \
            f"T={n_steps}"


def test_last_matches_final_slice():
    reduced, trace = _both({"last": last(X)})
    assert jnp.allclose(reduced["last"], trace["x"][-1], rtol=1e-6)


def test_running_max_and_min_match():
    """`x` rises monotonically, so the minimum is the FIRST post-round value. A
    reducer seeded with `readout(initial_state)` instead of an identity would
    return the pre-run value here and pass every max-only check — this is the
    regression guard for that."""
    reduced, trace = _both({"hi": running_max(MEAN_X), "lo": running_min(MEAN_X)})
    per_step = jnp.mean(trace["x"], axis=1)
    assert jnp.allclose(reduced["hi"], jnp.max(per_step), rtol=1e-6)
    assert jnp.allclose(reduced["lo"], jnp.min(per_step), rtol=1e-6)
    assert jnp.allclose(reduced["lo"], per_step[0], rtol=1e-6)   # not the init state


def test_running_var_matches_population_variance():
    """Welford vs jnp.var on the materialized series — the reason for Welford is
    float32 cancellation, so this must hold on a series far from zero."""
    reduced, trace = _both({"var": running_var(MEAN_X)})
    assert jnp.allclose(reduced["var"], jnp.var(jnp.mean(trace["x"], axis=1)),
                        rtol=1e-4)


def test_rate_matches_a_boolean_mean_over_both_axes():
    pred = lambda s: s.node_attrs["x"] > 1.0
    reduced, trace = _both({"rate": rate(pred)})
    assert jnp.allclose(reduced["rate"], jnp.mean((trace["x"] > 1.0).astype(jnp.float32)),
                        rtol=1e-5)


# --- the per-agent shape: O(N) carry replacing an O(T*N) trace -------------------

def test_per_agent_total_then_gini_matches():
    """The concentration-metric shape: accumulate (N,), reduce once at the end."""
    reduced, trace = _both({"g": per_agent_total(X, then=gini_of),
                            "vec": per_agent_total(X)})
    assert jnp.allclose(reduced["vec"], jnp.sum(trace["x"], axis=0), rtol=1e-5)
    assert jnp.allclose(reduced["g"], gini_of(jnp.sum(trace["x"], axis=0)), rtol=1e-4)


def test_per_agent_late_mean_then_gini_matches():
    reduced, trace = _both({"g": per_agent_late_mean(X, then=gini_of)})
    assert jnp.allclose(reduced["g"], gini_of(jnp.mean(_late(trace["x"]), axis=0)),
                        rtol=1e-4)


# --- plumbing -------------------------------------------------------------------

def test_reducers_compose_with_a_cheap_trace():
    """The intended usage: fold the agent axis away, keep the (T,) scalar series."""
    key = jr.PRNGKey(0)
    finals, trace, reduced = run_scan_reduce(
        _round, _init(key), 30, key, {"late": late_mean(MEAN_X)},
        trace_fn=lambda s: {"mean_x": MEAN_X(s)})
    assert trace["mean_x"].shape == (30,)          # (T,), not (T, N)
    assert reduced["late"].shape == ()
    assert jnp.allclose(reduced["late"], jnp.mean(_late(trace["mean_x"])), rtol=1e-5)


def test_batch_gives_one_value_per_seed():
    """The sweep path: reduced[name] is the per-seed vector, no trajectory kept."""
    keys = jr.split(jr.PRNGKey(0), 8)
    finals, trace, reduced = run_scan_reduce_batch(
        _round, _init, 25, keys, {"late": late_mean(MEAN_X)})
    assert trace is None
    assert reduced["late"].shape == (8,)
    assert bool(jnp.all(jnp.isfinite(reduced["late"])))
    # ...and it agrees with reducing each seed independently. The batch path
    # splits each key into init/run halves, so an unsplit comparison would be
    # comparing different worlds.
    per_seed = []
    for k in keys:
        k_init, k_run = jr.split(k)
        per_seed.append(run_scan_reduce(_round, _init(k_init), 25, k_run,
                                        {"late": late_mean(MEAN_X)})[2]["late"])
    assert jnp.allclose(reduced["late"], jnp.array(per_seed), rtol=1e-5)


def test_empty_reducers_is_legal():
    key = jr.PRNGKey(0)
    finals, trace, reduced = run_scan_reduce(_round, _init(key), 5, key, {})
    assert reduced == {}
    assert int(finals.global_attrs["step"]) == 5


def test_runs_under_jit():
    key = jr.PRNGKey(0)
    fn = jax.jit(lambda k: run_scan_reduce(_round, _init(k), 20, k,
                                           {"late": late_mean(MEAN_X)})[2])
    assert bool(jnp.isfinite(fn(key)["late"]))


def test_custom_reducer_protocol():
    """A user-written Reducer needs nothing but the three functions — verify the
    protocol is usable directly, not only via the catalog."""
    count_above = Reducer(
        init=lambda s, T: jnp.zeros((), jnp.int32),
        update=lambda acc, s, t, T: acc + jnp.sum(s.node_attrs["x"] > 1.0),
        finalize=lambda acc, T: acc,
    )
    reduced, trace = _both({"n": count_above})
    assert int(reduced["n"]) == int(jnp.sum(trace["x"] > 1.0))
