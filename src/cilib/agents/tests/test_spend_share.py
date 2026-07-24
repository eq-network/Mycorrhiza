"""Behavioral contract of the spend-share policy (the catalog's required test)."""
import jax.numpy as jnp
import jax.random as jr

from cilib.agents import SpendSharePolicy


def test_zero_noise_is_pure_preference_passthrough():
    pref = jnp.array([0.5, 0.3, 0.2, 0.0])
    action = SpendSharePolicy(noise=0.0)(pref, jr.PRNGKey(0))
    assert bool(jnp.all(action == pref))


def test_actions_never_negative_under_noise():
    pref = jnp.array([0.05, 0.0, 0.95])
    policy = SpendSharePolicy(noise=1.0)
    actions = jnp.stack([policy(pref, jr.PRNGKey(i)) for i in range(20)])
    assert bool(jnp.all(actions >= 0.0))
