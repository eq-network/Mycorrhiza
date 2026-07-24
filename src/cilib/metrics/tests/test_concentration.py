"""Behavioral tests for the concentration metric family (direction, not bit-exact numbers)."""
import jax.numpy as jnp

from cilib.metrics.families.concentration import gini_of, hhi_of


def test_gini_of_equal_split_is_zero():
    assert float(gini_of(jnp.full(10, 3.7))) < 1e-6


def test_gini_of_maximal_inequality_approaches_one():
    values = jnp.zeros(100).at[0].set(50.0)
    assert float(gini_of(values)) > 0.95


def test_hhi_of_equal_split_is_one_over_n():
    n = 8
    assert abs(float(hhi_of(jnp.full(n, 2.0))) - 1.0 / n) < 1e-6


def test_hhi_of_monopoly_is_one():
    values = jnp.zeros(8).at[3].set(42.0)
    assert abs(float(hhi_of(values)) - 1.0) < 1e-6
