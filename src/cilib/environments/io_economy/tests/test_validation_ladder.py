"""The validation ladder: the substrate must reproduce classical IO results before its
disempowerment twist is trusted. The Leontief anchors are exact (this substrate's fixed
point IS the textbook solution), which is the register's point — see
docs/model-register-design.md §4 and the colocated ASSUMPTIONS.md."""
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.io_economy import attribution_series, spectral_radius

NO_TWIST = dict(sub_onset=10_000, pref_noise=0.0, spend_noise=0.0)


def test_rebalancing_converges_to_leontief_fixed_point():
    """Classical anchor: the sequential rebalancing dynamic x <- Ax + d has the
    Leontief solution (I−A)⁻¹d as its fixed point, and starting AT that solution the
    trajectory is stationary (the Neumann-series identity, run as dynamics)."""
    env = make_env("io_economy", **NO_TWIST)
    _, trace = env.run(jr.PRNGKey(0), n_steps=60)
    A = trace["technical"][-1]
    d = trace["demand_h"][-1] + trace["demand_ai"][-1]
    x_star = jnp.linalg.solve(jnp.eye(A.shape[0]) - A, d)
    assert float(jnp.max(jnp.abs(trace["gross_output"][-1] - x_star))) < 1e-3
    assert float(jnp.max(jnp.abs(trace["gross_output"][-1]
                                 - trace["gross_output"][0]))) < 1e-3


def test_no_ai_attribution_is_one():
    """Before any recipe rewiring there is no AI demand: every unit of activity is
    ultimately attributable to human final demand."""
    env = make_env("io_economy", **NO_TWIST)
    _, trace = env.run(jr.PRNGKey(0), n_steps=60)
    share = attribution_series(trace)
    assert bool(jnp.all(share > 0.999))


def test_hypothetical_extraction_matches_analytic():
    """The instrument rung: shrinking human demand and measuring the output response
    (the counterfactual instrument's move) must recover the analytic IO answer
    1ᵀ(I−A)⁻¹Δd_H — hypothetical extraction in closed form. Paired same-key runs."""
    env1 = make_env("io_economy", **NO_TWIST)
    env2 = make_env("io_economy", init_income=0.8, **NO_TWIST)
    _, t1 = env1.run(jr.PRNGKey(0), n_steps=60)
    _, t2 = env2.run(jr.PRNGKey(0), n_steps=60)
    A = t1["technical"][-1]
    eye = jnp.eye(A.shape[0])
    measured = jnp.sum(t1["gross_output"][-1]) - jnp.sum(t2["gross_output"][-1])
    analytic = jnp.sum(jnp.linalg.solve(eye - A, t1["demand_h"][-1]
                                        - t2["demand_h"][-1]))
    assert abs(float(measured - analytic)) < 1e-2 * abs(float(analytic))


def test_substitution_decays_human_share_and_spectral_margin():
    """The twist, undefended: recipe rewiring (labor -> purchased AI cognition at cost
    parity) sends the wage bill toward zero, human demand attribution down with it,
    and the spectral margin toward the reproduction boundary — the last is the
    accounting identity itself (column sums climb as labor exits)."""
    env = make_env("io_economy")
    _, trace = env.run(jr.PRNGKey(0), n_steps=250)
    share = attribution_series(trace)
    assert bool(jnp.all(share[:20] > 0.99))
    assert float(share[-1]) < 0.5

    rho_early = spectral_radius(trace["technical"][0])
    rho_late = spectral_radius(trace["technical"][-1])
    assert float(rho_late) > float(rho_early) + 0.3
    assert float(rho_late) < 1.0                    # the loop stays convergent

    assert float(trace["wage_bill"][-1]) < 0.2 * float(trace["wage_bill"][0])


def test_full_reinvestment_is_relative_disempowerment():
    """Gradual Disempowerment §2 distinguishes absolute from relative disempowerment;
    ``reinvest_rate`` is the dial. At 1.0 demand is conserved: total activity GROWS
    while the human share collapses — the economy persists and runs for its own loop
    (the default 0.3 leaks demand and gives the absolute-collapse endpoint instead,
    covered by the previous rung)."""
    env = make_env("io_economy", reinvest_rate=1.0)
    _, trace = env.run(jr.PRNGKey(0), n_steps=250)
    total = jnp.sum(trace["gross_output"], axis=-1)
    share = attribution_series(trace)
    assert float(total[-1]) > float(total[0])            # activity persists (grows)
    assert float(share[-1]) < 0.05                       # humans exit
    ai_spend_late = jnp.mean(jnp.sum(trace["demand_ai"], axis=-1)[-50:])
    assert float(ai_spend_late) > 0.25 * float(total[-1])  # the loop runs the economy


def test_ai_revenue_tax_defends_human_share():
    """Defense rung: taxing the AI margin before it becomes self-directed demand and
    transferring it to households keeps human purchasing power — and with it the human
    demand share — alive. Same mechanism entry as compute_economy (cross-substrate
    reuse is the register's acceptance story)."""
    from cilib.core.schedule import scheduled
    from cilib.mechanisms.fiscal import AIRevenueTaxConfig, make_ai_revenue_tax

    tax = scheduled(make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=0.5)), onset=50)
    base = make_env("io_economy")
    defended = make_env("io_economy", mechanisms=(tax,))
    _, t_base = base.run(jr.PRNGKey(0), n_steps=250)
    _, t_def = defended.run(jr.PRNGKey(0), n_steps=250)
    share_base = float(jnp.mean(attribution_series(t_base)[-50:]))
    share_def = float(jnp.mean(attribution_series(t_def)[-50:]))
    assert share_def > share_base + 0.1
