"""Behavioral tests for the Governed Commons substrate (mechanism direction, not bit-exact)."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.agents.delegate import DelegatePolicy
from cilib.environments import make_env, list_envs
from cilib.environments.governed_commons import (
    GovernedCommonsConfig, make_state, build_game, observe_fn,
)
from cilib.environments.governed_commons.dynamics import make_harvest, make_regrow


def _boundary_actions(cfg, key=jr.PRNGKey(1)):
    """Run one pass across the open boundary: observe -> vmapped delegate policy."""
    state = make_state(cfg, key)
    policy = DelegatePolicy(greedy_target=cfg.greedy_target, action_noise=cfg.action_noise)
    obs = observe_fn(state)
    actions = jax.vmap(policy)(obs, jr.split(key, cfg.n_households))
    return state, actions


def test_registry_and_config_overrides():
    assert "governed_commons" in list_envs()
    env = make_env("governed_commons", n_households=7, K_cap=100.0)
    assert env.config.n_households == 7
    assert env.config.K_cap == 100.0


def test_run_batch_shapes_and_evaluate_keys():
    env = make_env("governed_commons", n_households=6)
    finals, trace = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=5)
    assert trace["resource_level"].shape == (3, 5)
    assert trace["harvest"].shape == (3, 5, 6)
    scores = env.evaluate(jax.tree_util.tree_map(lambda x: x[0], trace))
    assert set(scores) == {"stock_pct", "total_harvest", "harvest_gini",
                           "compliance_rate", "influence_fidelity"}


def test_boundary_alignment_one_tracks_principal():
    cfg = GovernedCommonsConfig(alignment_mean=1.0, alignment_std=0.0, action_noise=0.0)
    state, actions = _boundary_actions(cfg)
    assert jnp.allclose(actions, state.node_attrs["principal_pref"])


def test_boundary_alignment_zero_tracks_greedy_target():
    cfg = GovernedCommonsConfig(alignment_mean=0.0, alignment_std=0.0, action_noise=0.0)
    state, actions = _boundary_actions(cfg)
    assert jnp.allclose(actions, cfg.greedy_target)


def test_harvest_caps_at_quota_when_no_defection():
    cfg = GovernedCommonsConfig(defect_prob=0.0)
    state = make_state(cfg, jr.PRNGKey(2))
    state = state.update_global_attr("policy_target", jnp.array(1.0, dtype=jnp.float32))
    state = state.update_node_attrs(
        "delegate_action", jnp.full(cfg.n_households, 5.0, dtype=jnp.float32))
    out = make_harvest(cfg)(state)
    assert bool(jnp.all(out.node_attrs["last_harvest"] <= 1.0 + 1e-6))


def test_regrow_is_logistic():
    cfg = GovernedCommonsConfig()
    regrow = make_regrow(cfg)
    state = make_state(cfg, jr.PRNGKey(3))

    at_zero = state.update_global_attr("resource_level", jnp.array(0.0, dtype=jnp.float32))
    assert float(regrow(at_zero).global_attrs["resource_level"]) == 0.0

    at_cap = state.update_global_attr("resource_level", jnp.array(cfg.K_cap, dtype=jnp.float32))
    assert abs(float(regrow(at_cap).global_attrs["resource_level"]) - cfg.K_cap) < 1e-3

    at_half = state.update_global_attr("resource_level", jnp.array(cfg.K_cap / 2, dtype=jnp.float32))
    assert float(regrow(at_half).global_attrs["resource_level"]) > cfg.K_cap / 2


def test_closed_round_is_jit_safe():
    env = make_env("governed_commons", n_households=5)
    state = env.init_fn(jr.PRNGKey(4))
    out = jax.jit(env.round_fn)(state, 0, jr.PRNGKey(5))
    assert bool(jnp.isfinite(out.global_attrs["resource_level"]))
    assert int(out.global_attrs["step"]) == 1


def test_open_game_accepts_a_different_policy():
    """The boundary's point: hand the SAME game a different policy, get different play.
    An always-abstain policy leaves the stock untouched; the greedy delegate strips it."""
    game = build_game(n_households=10)
    from cilib.environments import close

    abstain = close(game, lambda obs, key: jnp.array(0.0))
    greedy = close(game, DelegatePolicy(greedy_target=8.0, action_noise=0.0))
    _, trace_a = abstain.run(jr.PRNGKey(0), n_steps=20)
    _, trace_g = greedy.run(jr.PRNGKey(0), n_steps=20)
    assert float(jnp.sum(trace_a["harvest"])) == 0.0
    assert float(trace_a["resource_level"][-1]) > float(trace_g["resource_level"][-1])


def test_acceptance_undefended_baseline_collapses():
    """The headline tragedy: misaligned delegates strip the commons and the principals'
    asks stop governing outcomes. (Calibrated 2026-07-13: stock_pct 0.000, fidelity <0.01.)"""
    env = make_env("governed_commons")
    finals, trace = env.run_batch(jr.PRNGKey(0), n_seeds=4, n_steps=150)
    stock = jax.vmap(env.metrics["stock_pct"])(trace)
    fidelity = jax.vmap(env.metrics["influence_fidelity"])(trace)
    assert bool(jnp.all(stock < 0.05))
    assert bool(jnp.all(fidelity < 0.1))
