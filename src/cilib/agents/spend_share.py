"""
Spend-share policy — the classical-ABM household consumption rule.

The demand-side sibling of ``labor_supply``: spend your income according to your
preference weights over sectors. No learning, no optimization — in the recipe economy
(``environments/io_economy``) the *preferences* are the human channel, and the question
is whether the economy's composition still follows them.
"""
import jax.numpy as jnp
from jax import random


class SpendSharePolicy:
    """``obs = (S,) spend-preference row`` → desired spending weights over sectors.

        weights = max(pref + noise, 0)

    Normalization to shares happens in the environment's ``step_fn`` (which also masks
    non-household rows), so the policy stays a pure per-agent map. ``noise`` = 0 gives
    pure preference pass-through — the deliberately transparent default.
    """

    def __init__(self, noise: float = 0.0):
        self.noise = noise

    def __call__(self, obs: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        return jnp.maximum(obs + self.noise * random.normal(key, obs.shape), 0.0)
