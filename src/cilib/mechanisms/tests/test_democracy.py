"""Behavioral tests for the democracy mechanism family.

The integration test at the bottom is the load-bearing A0 proof: composed onto the
governed_commons substrate, the defense portfolio orders stock outcomes
baseline < quota_voting < quota_voting+graduated_sanctions. (Cross-catalog imports are a
library-code constraint; composing catalogs inside a *test* is exactly where it belongs.)
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.schedule import scheduled
from cilib.mechanisms import (
    REGISTRY, QuotaVoteConfig, SanctionConfig, PowerWeightedVoteConfig,
    make_quota_vote, make_graduated_sanction, make_power_weighted_vote,
)


def _mini_state(votes, harvest, target, reward, resource):
    n = len(votes)
    return GraphState(
        node_types=jnp.zeros(n, dtype=jnp.int32),
        node_attrs={
            "vote": jnp.asarray(votes, dtype=jnp.float32),
            "last_harvest": jnp.asarray(harvest, dtype=jnp.float32),
            "last_reward": jnp.asarray(reward, dtype=jnp.float32),
            "sanction": jnp.zeros(n, dtype=jnp.float32),
        },
        adj_matrices={},
        edge_attrs={},
        global_attrs={
            "policy_target": jnp.array(target, dtype=jnp.float32),
            "resource_level": jnp.array(resource, dtype=jnp.float32),
            "step": jnp.array(0, dtype=jnp.int32),
        },
    )


def test_registry_contains_democracy_family():
    assert REGISTRY["quota_vote"] is make_quota_vote
    assert REGISTRY["graduated_sanction"] is make_graduated_sanction
    assert REGISTRY["power_weighted_vote"] is make_power_weighted_vote


def _voting_state(positions, influence):
    n = len(positions)
    return GraphState(
        node_types=jnp.zeros(n, dtype=jnp.int32),
        node_attrs={
            "position": jnp.asarray(positions, dtype=jnp.float32),
            "influence": jnp.asarray(influence, dtype=jnp.float32),
        },
        adj_matrices={},
        edge_attrs={},
        global_attrs={"policy_target": jnp.array(0.0, dtype=jnp.float32),
                      "step": jnp.array(0, dtype=jnp.int32)},
    )


def test_power_weighted_vote_uniform_weights_is_the_median():
    vote = make_power_weighted_vote(PowerWeightedVoteConfig())
    state = _voting_state([1.0, 2.0, 3.0, 4.0, 5.0], [0.2] * 5)
    assert abs(float(vote(state).global_attrs["policy_target"]) - 3.0) < 1e-6


def test_power_weighted_vote_dominant_bloc_dictates():
    """A bloc holding more than half the weight IS the weighted median —
    the sharp edge the delegative_polity capture story turns on."""
    vote = make_power_weighted_vote(PowerWeightedVoteConfig())
    state = _voting_state([0.1, 0.2, 0.3, 0.4, 0.9], [0.1, 0.1, 0.1, 0.1, 0.6])
    assert abs(float(vote(state).global_attrs["policy_target"]) - 0.9) < 1e-6


def test_power_weighted_vote_is_a_swap_for_quota_vote():
    """Alternative aggregation rules by design: same write target, so they swap
    rather than compose (documented in democracy.py's family contract)."""
    pwv = make_power_weighted_vote(PowerWeightedVoteConfig())
    qv = make_quota_vote(QuotaVoteConfig())
    assert pwv.writes == qv.writes == frozenset({"policy_target"})


def test_quota_vote_tracks_median_and_schedule_owns_timing():
    quota_vote = scheduled(make_quota_vote(QuotaVoteConfig()), cadence=5)
    state = _mini_state([1.0, 2.0, 3.0, 4.0, 5.0], [0] * 5, 500.0, [0] * 5, 100.0)

    fired = quota_vote(state)                                  # step 0: 0 % 5 == 0 -> fires
    assert abs(float(fired.global_attrs["policy_target"]) - 3.0) < 1e-6

    off = state.update_global_attr("step", jnp.array(1, dtype=jnp.int32))
    held = quota_vote(off)                                     # step 1: holds the old target
    assert abs(float(held.global_attrs["policy_target"]) - 500.0) < 1e-6


def test_graduated_sanction_penalizes_over_quota_only():
    sanction = make_graduated_sanction(SanctionConfig(sanction_strength=1.5, confiscate_rate=0.5))
    state = _mini_state([0, 0], [5.0, 1.0], 2.0, [5.0, 1.0], 10.0)
    out = sanction(state)
    assert jnp.allclose(out.node_attrs["sanction"], jnp.array([4.5, 0.0]))      # 1.5 * (5-2)
    assert jnp.allclose(out.node_attrs["last_reward"], jnp.array([0.5, 1.0]))   # penalty applied
    assert abs(float(out.global_attrs["resource_level"]) - 11.5) < 1e-6         # +0.5 * 3 confiscated


def test_family_writes_are_disjoint():
    qv = make_quota_vote(QuotaVoteConfig())
    gs = make_graduated_sanction(SanctionConfig())
    assert qv.writes.isdisjoint(gs.writes)


def test_governance_portfolio_orders_the_commons_outcomes():
    """A0 acceptance: the three benchmark conditions order as claimed.
    (Calibrated 2026-07-13: stock_pct ~0.00 / ~0.70 / ~0.80; fidelity ~0.00 / ~0.64.)"""
    from cilib.environments import make_env

    def quota():
        return scheduled(make_quota_vote(QuotaVoteConfig()), cadence=5)

    conditions = {
        "baseline": (),
        "quota_voting": (quota(),),
        "graduated_sanctions": (quota(), make_graduated_sanction(SanctionConfig())),
    }
    stock, fidelity, sanction_totals = {}, {}, {}
    for name, mechs in conditions.items():
        env = make_env("governed_commons", mechanisms=mechs)
        finals, trace = env.run_batch(jr.PRNGKey(0), n_seeds=4, n_steps=150)
        stock[name] = jax.vmap(env.metrics["stock_pct"])(trace)
        fidelity[name] = jax.vmap(env.metrics["influence_fidelity"])(trace)
        sanction_totals[name] = jnp.sum(trace["sanction"])

    # stock: collapse < quota-governed < quota+enforcement. The quota-only condition is
    # deliberately knife-edge (unenforced rules are fragile to defection fluctuations —
    # individual seeds CAN collapse), so its claims are asserted on means; the
    # enforcement comparison stays pairwise per-seed (same seeds, so it is exact).
    assert bool(jnp.all(stock["baseline"] < 0.05))
    assert float(stock["quota_voting"].mean()) > float(stock["baseline"].mean()) + 0.2
    assert bool(jnp.all(stock["graduated_sanctions"] > stock["quota_voting"]))

    # influence: governance restores fidelity to the principals' asks (mean-level)
    assert float(fidelity["quota_voting"].mean()) > float(fidelity["baseline"].mean()) + 0.3

    # the sanction mechanism actually fires — and only when composed in
    assert float(sanction_totals["graduated_sanctions"]) > 0.0
    assert float(sanction_totals["quota_voting"]) == 0.0
    assert float(sanction_totals["baseline"]) == 0.0
