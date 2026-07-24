"""
Labor-supply policy — the classical-ABM household decision rule.

The deliberately simple ("stupid first") rule for the economic scenarios: work your
preference, adjusted mildly by how today's wage compares to a reference wage. No
learning, no optimization — the mechanisms should do the work, not the agents.
"""
import jax.numpy as jnp
from jax import random


class LaborSupplyPolicy:
    """``obs = [work_pref, wage]`` → labor supplied this tick.

        labor = work_pref * (1 + elasticity * (wage - wage_ref) / wage_ref) + noise

    ``wage_elasticity`` = 0 gives perfectly inelastic supply (pure preference);
    small positive values give the textbook upward-sloping response. Clipped at 0.
    """

    def __init__(self, wage_elasticity: float = 0.3, wage_ref: float = 1.0,
                 noise: float = 0.0):
        self.wage_elasticity = wage_elasticity
        self.wage_ref = wage_ref
        self.noise = noise

    def __call__(self, obs: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        work_pref, wage = obs[0], obs[1]
        response = 1.0 + self.wage_elasticity * (wage - self.wage_ref) / self.wage_ref
        return jnp.maximum(work_pref * response + self.noise * random.normal(key), 0.0)
