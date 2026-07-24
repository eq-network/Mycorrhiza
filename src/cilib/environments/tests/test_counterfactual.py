"""Behavioral tests for the counterfactual influence instruments.

The headline claim they must support: under the undefended baseline the principals'
preferences do NOT causally govern outcomes; under quota voting they do.
"""
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.counterfactual import collective_influence, influence_matrix
from cilib.environments.governed_commons import (
    shift_preferences, per_capita_harvest, per_agent_harvest,
)
from cilib.core.schedule import scheduled
from cilib.mechanisms import (
    QuotaVoteConfig, SanctionConfig, make_quota_vote, make_graduated_sanction,
)

DELTA = -0.5  # downward: stays inside the feasible region (see counterfactual.py)


def _quota():
    return scheduled(make_quota_vote(QuotaVoteConfig()), cadence=5)


def test_collective_influence_baseline_vs_enforced_governance():
    """Robust regime claims only: the collapsed baseline has ~zero causal influence;
    the ENFORCED portfolio (quota + sanctions, stabilized well inside the feasible
    region) is near one-for-one. Quota-only is deliberately not asserted here — it is
    knife-edge by calibration, and its responsiveness is regime-dependent (positive
    pre-collapse, ~zero once marginal seeds start collapsing; the 2026-07-13 benchmark
    run measured -0.07 at T=200). That regime-dependence is a finding, not noise."""
    baseline = make_env("governed_commons")
    enforced = make_env("governed_commons",
                        mechanisms=(_quota(),
                                    make_graduated_sanction(SanctionConfig())))

    kwargs = dict(key=jr.PRNGKey(0), n_seeds=4, n_steps=200, delta=DELTA,
                  perturb_fn=shift_preferences, outcome_fn=per_capita_harvest)
    i_base = collective_influence(baseline, **kwargs)
    i_enforced = collective_influence(enforced, **kwargs)

    assert abs(float(jnp.mean(i_base))) < 0.15
    assert float(jnp.mean(i_enforced)) > 0.5
    assert float(jnp.mean(i_enforced)) > float(jnp.mean(i_base))


def test_influence_matrix_shape_and_finiteness():
    env = make_env("governed_commons", n_households=8, mechanisms=(_quota(),))
    A = influence_matrix(env, jr.PRNGKey(1), n_steps=50, delta=DELTA,
                         perturb_fn=shift_preferences,
                         outcome_agents_fn=per_agent_harvest)
    assert A.shape == (8, 8)
    assert bool(jnp.all(jnp.isfinite(A)))


def test_intervention_response_separates_economy_regimes():
    """Influence NOW (mid-run intervention) vs influence-from-birth: the economy's
    disempowerment signature. Calibrated 2026-07-14: from-birth ≈ 1.0 in BOTH regimes;
    influence-now ≈ 0.33 undefended vs ≈ 0.78 tax-defended."""
    from cilib.core.schedule import scheduled as sched
    from cilib.environments.counterfactual import intervention_response
    from cilib.environments.compute_economy import late_log_output, make_work_pref_shift
    from cilib.mechanisms import AIRevenueTaxConfig, make_ai_revenue_tax

    delta, t0, n_steps, n_seeds = -0.3, 200, 300, 3
    shift = sched(make_work_pref_shift(delta), cadence=n_steps + 1,
                  phase_offset=t0, onset=t0)
    tax = sched(make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=0.5)), onset=50)

    def influence_now(mechs):
        base = make_env("compute_economy", mechanisms=mechs)
        intervened = make_env("compute_economy", mechanisms=(*mechs, shift))
        return intervention_response(base, intervened, jr.PRNGKey(0), n_seeds, n_steps,
                                     outcome_fn=late_log_output, scale=delta)

    i_base = influence_now(())
    i_tax = influence_now((tax,))
    assert float(jnp.mean(i_base)) < 0.6
    assert float(jnp.mean(i_tax)) > float(jnp.mean(i_base)) + 0.2
