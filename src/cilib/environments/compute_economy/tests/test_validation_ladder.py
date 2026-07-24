"""The validation ladder: the substrate must reproduce classical results before its
disempowerment twist is trusted. (Calibrated 2026-07-14 — see config.py's docstring.)"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env


def _labor_share_series(trace):
    wage_bill = trace["wage"] * jnp.sum(trace["labor_supply"], axis=1)
    return wage_bill / jnp.maximum(trace["output"], 1e-6)


def test_cobb_douglas_limit_has_constant_labor_share():
    """Textbook: with unit elasticity of substitution the labor share is exactly α at
    every tick, no matter how large the compute stock grows."""
    env = make_env("compute_economy", rho=0.0, alpha=0.6)
    _, trace = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=200)
    share = jax.vmap(_labor_share_series)(trace)
    assert bool(jnp.all(jnp.abs(share - 0.6) < 1e-3))


def test_no_ai_arrivals_is_a_stable_steady_state():
    env = make_env("compute_economy", n_ai_slots=0, labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(1), n_steps=200)
    assert abs(float(trace["wage"][-1]) - float(trace["wage"][10])) < 1e-4
    assert float(jnp.sum(trace["capital"][-1])) == 0.0
    share = _labor_share_series(trace)
    assert bool(jnp.all(share > 0.99))                 # all income is labor income


def test_undefended_labor_share_decays():
    """The dependence-decay curve (Gradual Disempowerment §2): with σ = 2 substitutes,
    compounding AI compute drives the labor share from ~1 toward ~0.15."""
    env = make_env("compute_economy")                   # rho=0.5 default
    _, trace = env.run_batch(jr.PRNGKey(0), n_seeds=4, n_steps=300)
    share = jax.vmap(_labor_share_series)(trace)        # (S, T)
    assert bool(jnp.all(jnp.mean(share[:, :10], axis=1) > 0.9))
    assert bool(jnp.all(share[:, -1] < 0.35))


def test_labor_dependence_elasticity_recovers_cobb_douglas_alpha():
    """Instrument rung (2026-07-24): the labor-dependence elasticity — d log(output) /
    d log(labor) over the 10 ticks after a one-shot mid-run work-preference shift,
    paired same-key batches — must recover the CES identity. At ρ=0 the static output
    elasticity of labor IS α. (A long window fails this rung: the capital-path
    feedback adds ~+0.18 over 50 ticks — the reason the instrument uses a short
    post-intervention window.)"""
    from cilib.core.schedule import scheduled
    from cilib.environments.compute_economy import (
        make_work_pref_shift, make_window_log_output, make_window_log_labor,
    )
    from cilib.environments.counterfactual import intervention_elasticity

    T = 150
    t0 = 2 * T // 3
    shift = scheduled(make_work_pref_shift(-0.3),
                      cadence=T + 1, phase_offset=t0, onset=t0)
    base = make_env("compute_economy", rho=0.0, alpha=0.6)
    intervened = make_env("compute_economy", rho=0.0, alpha=0.6, mechanisms=(shift,))
    e = intervention_elasticity(base, intervened, jr.PRNGKey(0), 3, T,
                                outcome_fn=make_window_log_output(t0 + 1, 10),
                                channel_fn=make_window_log_labor(t0 + 1, 10))
    # Residual bias is strictly upward (capital falls WITH labor): ≈ +0.05 at w=10,
    # shrinking with the window. Bound both sides asymmetrically around α = 0.6.
    assert bool(jnp.all(e > 0.59)) and bool(jnp.all(e < 0.68))


def test_reinvestment_concentrates_income():
    """Rich-get-richer: compounding reinvestment concentrates income far beyond the
    no-AI economy, and concentration rises within the run."""
    baseline = make_env("compute_economy")
    no_ai = make_env("compute_economy", n_ai_slots=0)
    _, trace_b = baseline.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=300)
    _, trace_n = no_ai.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=300)

    gini_b = jax.vmap(baseline.metrics["income_gini"])(trace_b)
    gini_n = jax.vmap(no_ai.metrics["income_gini"])(trace_n)
    assert bool(jnp.all(gini_b > gini_n + 0.3))

    from cilib.metrics.families.concentration import hhi_of
    early = jax.vmap(lambda t: hhi_of(t["last_reward"][30]))(trace_b)
    late = jax.vmap(lambda t: hhi_of(t["last_reward"][290]))(trace_b)
    assert bool(jnp.all(late > early))
