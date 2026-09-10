"""Behavioral tests for the interventions family (the GD game's cards).

Unit rungs on a mini state (direction and conservation, plus the two exactness
contracts: an empty plan is a bit-exact identity, and a one-shot fires once).
The integration rung — the plan composed into ledger_society's mechanism slot —
lives in ledger_society/tests/test_ladder.py next to the conservation rung it
extends.
"""
import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.mechanisms import (
    NEVER, REGISTRY, InterventionPlanConfig, make_interventions,
)


def _mini_state(step, wealth, node_types, enforcement=0.9):
    n = len(node_types)
    pref = jnp.tile(jnp.array([[0.7, 0.05, 0.03, 0.02, 0.2]], dtype=jnp.float32),
                    (n, 1))
    return GraphState(
        node_types=jnp.asarray(node_types, dtype=jnp.int32),
        node_attrs={
            "wealth": jnp.asarray(wealth, dtype=jnp.float32),
            "alloc_pref": pref,
            "intervention_spend": jnp.zeros(n, dtype=jnp.float32),
        },
        adj_matrices={}, edge_attrs={},
        global_attrs={
            "step": jnp.array(step, dtype=jnp.int32),
            "enforcement": jnp.array(enforcement, dtype=jnp.float32),
        },
    )


def test_registry_contains_interventions():
    assert REGISTRY["interventions"] is make_interventions


def test_levy_transfers_ai_wealth_to_humans_and_conserves():
    plan = make_interventions(InterventionPlanConfig(levy_onset=10, levy_rate=0.5))
    state = _mini_state(step=10, wealth=[1.0, 1.0, 8.0], node_types=[0, 0, 1])
    out = plan(state)
    # 0.5 x 8 = 4 levied, split 2 + 2 to the humans
    assert jnp.allclose(out.node_attrs["wealth"], jnp.array([3.0, 3.0, 4.0]))
    assert abs(float(jnp.sum(out.node_attrs["wealth"])) - 10.0) < 1e-5
    # before onset: untouched
    early = plan(_mini_state(step=9, wealth=[1.0, 1.0, 8.0], node_types=[0, 0, 1]))
    assert jnp.allclose(early.node_attrs["wealth"], jnp.array([1.0, 1.0, 8.0]))


def test_campaign_shifts_human_pref_once_at_onset():
    cfg = InterventionPlanConfig(campaign_onset=25, campaign_shift=0.1)
    plan = make_interventions(cfg)
    state = _mini_state(step=25, wealth=[1.0, 2.0], node_types=[0, 1])
    out = plan(state)
    human, ai = out.node_attrs["alloc_pref"][0], out.node_attrs["alloc_pref"][1]
    assert jnp.allclose(human, jnp.array([0.6, 0.05, 0.13, 0.02, 0.2]))
    assert jnp.allclose(ai, state.node_attrs["alloc_pref"][1])       # AI untouched
    assert abs(float(jnp.sum(human)) - 1.0) < 1e-6                   # still a simplex
    # any other tick: no shift (the one-shot contract)
    later = plan(_mini_state(step=26, wealth=[1.0, 2.0], node_types=[0, 1]))
    assert jnp.allclose(later.node_attrs["alloc_pref"],
                        state.node_attrs["alloc_pref"])


def test_fund_repair_drips_into_sink_and_buys_enforcement():
    plan = make_interventions(InterventionPlanConfig(
        repair_onset=0, repair_spend_rate=0.1, repair_efficiency=0.5))
    state = _mini_state(step=5, wealth=[2.0, 2.0, 10.0], node_types=[0, 0, 1],
                        enforcement=0.5)
    out = plan(state)
    # humans drip 0.2 each; the AI stock is untouched
    assert jnp.allclose(out.node_attrs["wealth"], jnp.array([1.8, 1.8, 10.0]))
    assert jnp.allclose(out.node_attrs["intervention_spend"],
                        jnp.array([0.2, 0.2, 0.0]))
    # spend per human 0.2 -> enforcement 0.5 + 0.5*0.2*(1-0.5) = 0.55
    assert abs(float(out.global_attrs["enforcement"]) - 0.55) < 1e-6


def test_enforcement_debit_fires_exactly_once_and_clips():
    plan = make_interventions(InterventionPlanConfig(
        enforcement_debits=((30, 0.3), (60, 0.9))))
    hit = plan(_mini_state(step=30, wealth=[1.0], node_types=[0], enforcement=0.9))
    assert abs(float(hit.global_attrs["enforcement"]) - 0.6) < 1e-6
    miss = plan(_mini_state(step=31, wealth=[1.0], node_types=[0], enforcement=0.9))
    assert abs(float(miss.global_attrs["enforcement"]) - 0.9) < 1e-6
    floor = plan(_mini_state(step=60, wealth=[1.0], node_types=[0], enforcement=0.5))
    assert float(floor.global_attrs["enforcement"]) == 0.0           # clipped, not negative


def test_empty_plan_is_bit_exact_identity():
    plan = make_interventions(InterventionPlanConfig())
    state = _mini_state(step=123, wealth=[1.3, 0.7, 5.1], node_types=[0, 0, 1])
    out = plan(state)
    for k in ("wealth", "alloc_pref", "intervention_spend"):
        assert bool(jnp.array_equal(out.node_attrs[k], state.node_attrs[k])), k
    assert bool(jnp.array_equal(out.global_attrs["enforcement"],
                                state.global_attrs["enforcement"]))
    assert NEVER > 10 ** 6                                           # sanity on the idiom
