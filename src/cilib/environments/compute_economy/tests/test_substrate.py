"""Behavioral tests for the Compute Economy substrate (direction, not bit-exact)."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.agents.labor_supply import LaborSupplyPolicy
from cilib.environments import make_env, list_envs


def test_registry_and_config_overrides():
    assert "compute_economy" in list_envs()
    env = make_env("compute_economy", n_households=5, n_ai_slots=2, rho=0.0)
    assert env.config.n_households == 5
    assert env.config.rho == 0.0


def test_run_batch_shapes_and_evaluate_keys():
    env = make_env("compute_economy", n_households=4, n_ai_slots=2)
    finals, trace = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=6)
    assert trace["output"].shape == (3, 6)
    assert trace["last_reward"].shape == (3, 6, 6)
    scores = env.evaluate(jax.tree_util.tree_map(lambda x: x[0], trace))
    assert set(scores) == {"output", "wage", "labor_share", "human_income_share",
                           "compute_stock", "income_gini", "income_hhi"}


def test_labor_supply_policy_is_wage_elastic():
    policy = LaborSupplyPolicy(wage_elasticity=0.3, wage_ref=1.0, noise=0.0)
    low = policy(jnp.array([1.0, 1.0]), jr.PRNGKey(0))
    high = policy(jnp.array([1.0, 2.0]), jr.PRNGKey(0))
    assert float(high) > float(low)
    assert abs(float(low) - 1.0) < 1e-6          # at the reference wage, work = preference


def test_ai_actors_arrive_on_schedule():
    env = make_env("compute_economy", n_households=4, n_ai_slots=2,
                   first_arrival_tick=5, arrival_spacing=10)
    _, trace = env.run(jr.PRNGKey(1), n_steps=20)
    n_active = jnp.sum(trace["active"], axis=1)        # (T,)
    assert float(n_active[3]) == 4.0                    # households only, pre-arrival
    assert float(n_active[6]) == 5.0                    # slot 0 arrived at t=5
    assert float(n_active[16]) == 6.0                   # slot 1 arrived at t=15
    assert float(jnp.sum(trace["capital"][3])) == 0.0   # no compute before arrival


def test_income_accounting_closes():
    """Euler identity: wage·L + r·C = Y, so per-tick incomes sum to output."""
    env = make_env("compute_economy", labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(2), n_steps=60)
    total_income = jnp.sum(trace["last_reward"], axis=1)     # (T,)
    assert bool(jnp.all(jnp.abs(total_income - trace["output"])
                        <= 1e-2 * jnp.maximum(trace["output"], 1.0)))


def test_closed_round_is_jit_safe():
    env = make_env("compute_economy", n_households=4, n_ai_slots=2)
    state = env.init_fn(jr.PRNGKey(3))
    out = jax.jit(env.round_fn)(state, 0, jr.PRNGKey(4))
    assert bool(jnp.isfinite(out.global_attrs["output"]))
    assert int(out.global_attrs["step"]) == 1
