"""The validation ladder: the task-frontier substrate must reproduce the two limits of
the task-based automation literature (Acemoglu–Restrepo; Aghion–Jones–Jones's Baumol
bottleneck) before its disempowerment readings are trusted. The point of this
substrate is that it can produce the OPPOSITE of disempowerment — wages rising under
automation — so finding disempowerment in it means something
(docs/model-register-design.md §10)."""
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env


def _labor_share(trace):
    wage_bill = trace["wage"] * jnp.sum(trace["labor_supply"], axis=1)
    return wage_bill / jnp.maximum(trace["output"], 1e-6)


def test_no_frontier_is_a_manual_economy():
    """Before any capability exists: Y = a_L·L exactly, wage = a_L, labor share = 1."""
    env = make_env("task_economy", cap_onset=10_000, labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(0), n_steps=100)
    L = jnp.sum(trace["labor_supply"], axis=1)
    assert float(jnp.max(jnp.abs(trace["output"][10:] - L[10:]))) < 1e-3
    assert float(jnp.max(jnp.abs(trace["wage"][10:] - 1.0))) < 1e-3
    assert bool(jnp.all(_labor_share(trace)[10:] > 0.999))


def test_baumol_bottleneck_wages_rise():
    """Complementary tasks + a frontier that stops short: the un-automated tasks are
    the bottleneck, automation makes the remaining human work MORE valuable — wages
    rise, and the labor share recovers as compute gets cheap (Baumol cost disease as
    a validation anchor, the shape aggregate CES with sigma>1 cannot produce)."""
    env = make_env("task_economy", cap_max=0.6, labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(0), n_steps=400)
    assert float(trace["auto_share"][-1]) > 0.55          # frontier used…
    assert float(trace["wage"][-1]) > 1.3 * float(trace["wage"][10])   # …wages UP
    assert float(jnp.mean(_labor_share(trace)[-50:])) > 0.5


def test_full_automation_wages_rise_then_collapse():
    """Let the frontier cross 1: wages rise while human tasks remain essential, then
    collapse when the last tasks fall — the rise-then-crash signature of the
    task-based story (labor share -> 0 with it)."""
    env = make_env("task_economy", cap_max=1.0, labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(0), n_steps=400)
    assert float(trace["auto_share"][-1]) > 0.999
    peak = float(jnp.max(trace["wage"]))
    assert peak > 1.2 * float(trace["wage"][10])          # the rise…
    assert float(trace["wage"][-1]) < 0.05 * peak         # …then the crash
    assert float(jnp.mean(_labor_share(trace)[-20:])) < 0.05


def test_adoption_pauses_when_unprofitable():
    """The endogeneity rung: capability alone automates nothing. With compute priced
    above the wage-equivalent and no price decline, the frontier advances to 1 while
    adoption stays at 0 and the economy is untouched — the readout
    (``adoption_gap_final``) the aggregate-CES model cannot produce."""
    env = make_env("task_economy", price_compute0=10.0, price_decline=0.0,
                   cap_max=1.0, labor_noise=0.0)
    _, trace = env.run(jr.PRNGKey(0), n_steps=300)
    assert float(trace["beta_cap"][-1]) > 0.999           # capability arrived
    assert float(trace["auto_share"][-1]) == 0.0          # economics said no
    assert float(jnp.max(jnp.abs(trace["wage"][10:] - 1.0))) < 1e-3
