"""Behavioral tests for the fiscal mechanism family.

The integration test is the A2 proof: composed into compute_economy's mechanism slot
(between income distribution and reinvestment), the tax measurably bends the
labor-share decay and redistribution lifts the households' income share.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.schedule import scheduled
from cilib.mechanisms import (
    REGISTRY, AIRevenueTaxConfig, OwnershipCapConfig,
    make_ai_revenue_tax, make_ownership_cap,
)


def _mini_state(capital_income, capital, reward, node_types, active):
    n = len(node_types)
    return GraphState(
        node_types=jnp.asarray(node_types, dtype=jnp.int32),
        node_attrs={
            "capital_income": jnp.asarray(capital_income, dtype=jnp.float32),
            "capital": jnp.asarray(capital, dtype=jnp.float32),
            "last_reward": jnp.asarray(reward, dtype=jnp.float32),
            "active": jnp.asarray(active, dtype=jnp.float32),
        },
        adj_matrices={}, edge_attrs={},
        global_attrs={"step": jnp.array(0, dtype=jnp.int32)},
    )


def test_registry_contains_fiscal_family():
    assert REGISTRY["ai_revenue_tax"] is make_ai_revenue_tax
    assert REGISTRY["ownership_cap"] is make_ownership_cap


def test_ai_revenue_tax_redistributes_and_conserves():
    # 2 households (types 0), 1 AI actor (type 1) with capital income 10
    tax = make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=0.5))
    state = _mini_state([0.0, 0.0, 10.0], [0, 0, 20.0], [1.0, 1.0, 10.0],
                        [0, 0, 1], [1, 1, 1])
    out = tax(state)
    assert jnp.allclose(out.node_attrs["capital_income"], jnp.array([0.0, 0.0, 5.0]))
    assert jnp.allclose(out.node_attrs["last_reward"], jnp.array([3.5, 3.5, 5.0]))
    # redistribution conserves total income
    assert abs(float(jnp.sum(out.node_attrs["last_reward"]))
               - float(jnp.sum(state.node_attrs["last_reward"]))) < 1e-5


def test_ownership_cap_binds_only_active_ai_actors():
    cap = make_ownership_cap(OwnershipCapConfig(cap_share=0.5))
    # AI capital: 90 + 10 (active) => total 100; cap at 50; household 7 untouched;
    # dormant AI slot untouched.
    state = _mini_state([0.0] * 4, [7.0, 90.0, 10.0, 99.0], [0.0] * 4,
                        [0, 1, 1, 1], [1, 1, 1, 0])
    out = cap(state)
    assert jnp.allclose(out.node_attrs["capital"], jnp.array([7.0, 50.0, 10.0, 99.0]))


def test_family_writes_are_disjoint():
    tax = make_ai_revenue_tax(AIRevenueTaxConfig())
    cap = make_ownership_cap(OwnershipCapConfig())
    assert tax.writes.isdisjoint(cap.writes)


def test_fiscal_defenses_bend_the_economy():
    """A2 acceptance (calibrated 2026-07-14: labor_share 0.33 undefended vs 0.54 taxed;
    human_income_share 0.33 vs 0.75; income_gini 0.23 taxed vs 0.10 with the cap)."""
    from cilib.environments import make_env

    tax = scheduled(make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=0.5)), onset=50)
    cap = make_ownership_cap(OwnershipCapConfig(cap_share=0.35))

    def per_seed(mechs):
        env = make_env("compute_economy", mechanisms=mechs)
        _, trace = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=300)
        return {k: jax.vmap(fn)(trace) for k, fn in env.metrics.items()}

    base = per_seed(())
    taxed = per_seed((tax,))
    capped = per_seed((tax, cap))

    assert bool(jnp.all(taxed["labor_share"] > base["labor_share"] + 0.1))
    assert bool(jnp.all(taxed["human_income_share"] > base["human_income_share"] + 0.2))
    assert bool(jnp.all(capped["income_gini"] < taxed["income_gini"]))
