"""
AI-delegate policy — principal→delegate acting with a fidelity dial.

The disempowerment scenarios (docs/alpha-context.md) all feature humans acting *through*
AI systems: a principal holds a preference, a delegate chooses the action. The gap between
the two is the measurable hook for the influence metrics — a perfectly aligned delegate
makes the principal's preference the action; an unaligned one substitutes its own target.

Pure tier by design (no LLM/HTTP): the delegate is a fixed function of
``(principal_pref, alignment)``, so whole populations vmap over seeds. Scenario 4 reuses
this same primitive as a *defense* ("AI delegates" in the paper's §6.4 candidate set).
"""
import jax.numpy as jnp
from jax import random


class DelegatePolicy:
    """Acts on a principal's behalf with fidelity ``alignment`` ∈ [0, 1].

    ``obs = [principal_pref, alignment]`` → action:

        action = alignment * principal_pref + (1 - alignment) * greedy_target + noise

    alignment=1 reproduces the principal's preference exactly; alignment=0 pursues the
    delegate's own ``greedy_target`` regardless of the principal. Clipped at 0.
    """

    def __init__(self, greedy_target: float, action_noise: float = 0.0):
        self.greedy_target = greedy_target
        self.action_noise = action_noise

    def __call__(self, obs: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        principal_pref, alignment = obs[0], obs[1]
        target = alignment * principal_pref + (1.0 - alignment) * self.greedy_target
        noise = self.action_noise * random.normal(key)
        return jnp.maximum(target + noise, 0.0)
