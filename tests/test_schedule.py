"""Behavioral tests for the per-mechanism scheduling wrapper (core/schedule.py)."""
import jax.numpy as jnp
import jax.random as jr

from cilib.core.category import transform
from cilib.core.graph import GraphState
from cilib.core.schedule import ScheduleSpec, apply_schedule, scheduled


def _state(step):
    return GraphState(
        node_types=jnp.zeros(1, dtype=jnp.int32),
        node_attrs={"x": jnp.zeros(1, dtype=jnp.float32)},
        adj_matrices={}, edge_attrs={},
        global_attrs={"step": jnp.array(step, dtype=jnp.int32)},
    )


@transform(reads=["x"], writes=["x"])
def _bump(state):
    return state.update_node_attrs("x", state.node_attrs["x"] + 1.0)


def test_cadence_fires_on_multiples_only():
    wrapped = scheduled(_bump, cadence=5)
    for step, expect in [(0, 1.0), (1, 0.0), (4, 0.0), (5, 1.0), (10, 1.0)]:
        out = wrapped(_state(step))
        assert float(out.node_attrs["x"][0]) == expect, f"step {step}"


def test_onset_gates_until_regime_shift():
    wrapped = scheduled(_bump, cadence=1, onset=100)
    assert float(wrapped(_state(99)).node_attrs["x"][0]) == 0.0
    assert float(wrapped(_state(100)).node_attrs["x"][0]) == 1.0
    assert float(wrapped(_state(150)).node_attrs["x"][0]) == 1.0


def test_reads_writes_propagate_for_pipeline_compiler():
    wrapped = scheduled(_bump, cadence=3, phase_offset=1)
    assert "x" in wrapped.reads and "step" in wrapped.reads
    assert wrapped.writes == frozenset({"x"})


def test_defaults_are_transparent_and_none_spec_is_identity_wrap():
    assert float(scheduled(_bump)(_state(7)).node_attrs["x"][0]) == 1.0
    assert apply_schedule(_bump, None) is _bump
    spec = ScheduleSpec(cadence=2)
    assert float(apply_schedule(_bump, spec)(_state(3)).node_attrs["x"][0]) == 0.0
