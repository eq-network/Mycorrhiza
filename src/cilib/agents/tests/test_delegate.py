"""Behavioral tests for the ai_delegate catalog entry."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.agents import REGISTRY, DelegatePolicy


def test_registry_contains_ai_delegate():
    assert REGISTRY["ai_delegate"] is DelegatePolicy


def test_alignment_one_tracks_principal_exactly():
    policy = DelegatePolicy(greedy_target=8.0, action_noise=0.0)
    obs = jnp.array([2.5, 1.0])  # [principal_pref, alignment]
    assert float(policy(obs, jr.PRNGKey(0))) == 2.5


def test_alignment_zero_tracks_greedy_target_exactly():
    policy = DelegatePolicy(greedy_target=8.0, action_noise=0.0)
    obs = jnp.array([2.5, 0.0])
    assert float(policy(obs, jr.PRNGKey(0))) == 8.0


def test_vmap_safe_over_population():
    policy = DelegatePolicy(greedy_target=8.0, action_noise=0.3)
    n = 16
    obs = jnp.stack([jnp.linspace(0.5, 3.0, n), jnp.linspace(0.0, 1.0, n)], axis=1)
    keys = jr.split(jr.PRNGKey(1), n)
    actions = jax.vmap(policy)(obs, keys)
    assert actions.shape == (n,)
    assert bool(jnp.all(actions >= 0.0))
