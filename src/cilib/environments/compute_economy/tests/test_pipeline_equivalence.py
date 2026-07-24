"""compile_pipeline == sequential program order for the economy's post-action pipeline."""
import jax.numpy as jnp
import jax.random as jr

from cilib.core.category import sequential
from cilib.core.pipeline import compile_pipeline
from cilib.core.schedule import scheduled
from cilib.environments.compute_economy import ComputeEconomyConfig, make_state, build_steps
from cilib.mechanisms import (
    AIRevenueTaxConfig, OwnershipCapConfig, make_ai_revenue_tax, make_ownership_cap,
)


def _acting_state(cfg, key=jr.PRNGKey(0)):
    state = make_state(cfg, key)
    labor = jnp.concatenate([jnp.linspace(0.5, 1.5, cfg.n_households),
                             jnp.zeros(cfg.n_ai_slots)]).astype(jnp.float32)
    return state.update_node_attrs("labor_supply", labor)


def _run_both(steps, state, n_rounds=30):
    seq = sequential(*steps)
    comp = compile_pipeline(list(steps))
    s_seq, s_comp = state, state
    for _ in range(n_rounds):
        s_seq, s_comp = seq(s_seq), comp(s_comp)
    return s_seq, s_comp


def _assert_states_match(a, b):
    for name in a.node_attrs:
        assert jnp.allclose(a.node_attrs[name], b.node_attrs[name]), name
    for name in a.global_attrs:
        if isinstance(a.global_attrs[name], jnp.ndarray):
            assert jnp.allclose(a.global_attrs[name], b.global_attrs[name]), name


def test_compiled_equals_sequential_no_mechanisms():
    cfg = ComputeEconomyConfig(n_households=6, n_ai_slots=2, first_arrival_tick=5)
    steps = build_steps(cfg)
    _assert_states_match(*_run_both(steps, _acting_state(cfg)))


def test_compiled_equals_sequential_with_fiscal_mechanisms():
    """Also proves the fiscal family splices into the slot without cycle or conflict."""
    cfg = ComputeEconomyConfig(n_households=6, n_ai_slots=2, first_arrival_tick=5)
    mechs = (scheduled(make_ai_revenue_tax(AIRevenueTaxConfig()), onset=10),
             make_ownership_cap(OwnershipCapConfig()))
    steps = build_steps(cfg, mechs)
    _assert_states_match(*_run_both(steps, _acting_state(cfg)))
