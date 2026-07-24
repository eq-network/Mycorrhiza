"""compile_pipeline must be numerically identical to sequential program order —
the CLAUDE.md verification rule, made executable for this substrate's post-action pipeline."""
import jax.numpy as jnp
import jax.random as jr

from cilib.core.category import sequential
from cilib.core.pipeline import compile_pipeline
from cilib.core.schedule import scheduled
from cilib.environments.governed_commons import GovernedCommonsConfig, make_state, build_steps
from cilib.mechanisms import (
    QuotaVoteConfig, SanctionConfig, make_quota_vote, make_graduated_sanction,
)


def _acting_state(cfg, key=jr.PRNGKey(0)):
    """A state mid-round: actions already applied (the pipeline runs post-boundary)."""
    state = make_state(cfg, key)
    desired = jnp.linspace(0.5, 6.0, cfg.n_households).astype(jnp.float32)
    return state.update_node_attrs("delegate_action", desired)


def _run_both(steps, state, n_rounds=5):
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
    cfg = GovernedCommonsConfig(n_households=8)
    steps = build_steps(cfg)
    _assert_states_match(*_run_both(steps, _acting_state(cfg)))


def test_compiled_equals_sequential_with_mechanisms():
    """Also proves the democracy mechanisms splice in without a cycle or write conflict."""
    cfg = GovernedCommonsConfig(n_households=8)
    mechs = (scheduled(make_quota_vote(QuotaVoteConfig()), cadence=5),
             make_graduated_sanction(SanctionConfig()))
    steps = build_steps(cfg, mechs)
    _assert_states_match(*_run_both(steps, _acting_state(cfg)))
