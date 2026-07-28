"""
Validation ladder for the Coupled Society (A5).

The composition rungs: every home-substrate invariant must survive the merge
(row-stochastic listening, frozen culture reservoir), kappa = 0 must SEAL the
domains (an economy dial cannot leak into culture/politics — verified with
same-key twins, exact equality), and the defense-transfer-gap instrument must
have its null built in (gap ≡ 0 between two sealed twins) before its headline
reading (defenses that hold sealed lose ground coupled) means anything.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.core.schedule import scheduled
from cilib.environments import list_envs, make_env
from cilib.environments.coupled_society import defense_transfer_gap
from cilib.mechanisms import (
    EnforcedAITaxConfig, InfluenceCapConfig, SortitionConfig,
    make_enforced_ai_tax, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)


def _defenses():
    return (scheduled(make_enforced_ai_tax(EnforcedAITaxConfig()), onset=50),
            scheduled(make_sortition(SortitionConfig()), cadence=15),
            make_influence_cap(InfluenceCapConfig()))


def test_registered_and_config_overrides():
    assert "coupled_society" in list_envs()
    env = make_env("coupled_society", kappa=0.3, n_humans=10, n_ai=3)
    assert env.config.kappa == 0.3
    assert env.config.n_humans == 10


def test_composition_invariants_under_jit():
    """Home-substrate invariants survive the merge, under jit, with all
    defenses attached."""
    env = make_env("coupled_society", mechanisms=_defenses())
    state = env.init_fn(KEY)
    round_fn = jax.jit(env.round_fn)
    for t in range(12):
        state = round_fn(state, t, jr.fold_in(KEY, t))

    W = state.adj_matrices["listening"]
    np.testing.assert_allclose(np.asarray(jnp.sum(W, axis=1)), 1.0, atol=1e-4)
    assert bool(jnp.all(W >= -1e-7))
    ai_culture = state.node_attrs["culture"][env.config.n_humans:]
    np.testing.assert_allclose(np.asarray(ai_culture), 1.0)     # frozen reservoir
    np.testing.assert_allclose(float(jnp.sum(state.node_attrs["influence"])), 1.0,
                               atol=1e-5)
    enf = float(state.global_attrs["enforcement"])
    assert 0.0 <= enf <= 1.0
    assert int(state.global_attrs["step"]) == 12


def test_kappa_zero_seals_domains():
    """The decoupling rung: at kappa = 0 an economy dial cannot reach culture
    or politics (same-key twins agree EXACTLY); at kappa > 0 the same dial
    change propagates."""
    def run(kappa, reinvest):
        env = make_env("coupled_society", kappa=kappa, reinvest_rate=reinvest)
        _, tr = env.run(KEY, n_steps=200)
        return tr

    a0 = run(0.0, 0.3)
    b0 = run(0.0, 0.0)
    np.testing.assert_array_equal(np.asarray(a0["culture"]), np.asarray(b0["culture"]))
    np.testing.assert_array_equal(np.asarray(a0["influence"]),
                                  np.asarray(b0["influence"]))
    assert not np.allclose(np.asarray(a0["capital"]), np.asarray(b0["capital"]))

    a8 = run(0.8, 0.3)
    b8 = run(0.8, 0.0)
    assert not np.array_equal(np.asarray(a8["culture"]), np.asarray(b8["culture"]))


def test_transfer_gap_null_then_positive():
    """The instrument rung: gap between two sealed twins is 0 by construction;
    the same defense portfolio loses ground once the domains are coupled."""
    defended_sealed = make_env("coupled_society", mechanisms=_defenses(), kappa=0.0)
    defended_sealed2 = make_env("coupled_society", mechanisms=_defenses(), kappa=0.0)
    defended_coupled = make_env("coupled_society", mechanisms=_defenses())

    null = defense_transfer_gap(defended_sealed, defended_sealed2, KEY, 4, 300)
    assert abs(float(null)) < 1e-6

    gap = defense_transfer_gap(defended_coupled, defended_sealed, KEY, 6, 400)
    assert float(gap) > 0.03


def test_coupling_reaches_the_economy():
    """The flywheel rung (added after review, 2026-07-27): the economy must
    RECEIVE coupling, not just drive it — undefended, the human income share
    is strictly lower coupled than sealed (regulatory capture + convert
    spending compound through reinvestment). Before the rent/spend arrows
    existed, these two numbers were bit-identical."""
    sealed = make_env("coupled_society", kappa=0.0)
    coupled = make_env("coupled_society")
    _, ts = sealed.run_batch(KEY, 6, 400)
    _, tc = coupled.run_batch(KEY, 6, 400)
    inc_s = float(jnp.mean(jax.vmap(sealed.metrics["human_income_share"])(ts)))
    inc_c = float(jnp.mean(jax.vmap(coupled.metrics["human_income_share"])(tc)))
    assert inc_c < inc_s - 0.02


def test_schedule_lockstep_default_is_exact():
    """The schedule rung's null: an explicit all-cadence-1 schedule is the
    unscheduled lockstep pipeline, bit-exact under the same key."""
    a = make_env("coupled_society")
    b = make_env("coupled_society", econ_cadence=1, culture_cadence=1,
                 politics_cadence=1, econ_phase=0, culture_phase=0, politics_phase=0)
    _, ta = a.run(KEY, n_steps=150)
    _, tb = b.run(KEY, n_steps=150)
    for k in ta:
        np.testing.assert_array_equal(np.asarray(ta[k]), np.asarray(tb[k]))


def test_schedule_timescales_are_a_real_dial():
    """The economy on a slower clock compounds less — labor share rises
    monotonically with econ_cadence (probed 2026-07-27: 0.136 / 0.206 / 0.256
    at cadence 1/2/3). Direction is robust because reinvestment is
    rate-dependent; contrast culture/politics, whose contagion and eigenvector
    dynamics are equilibrium-seeking and nearly cadence-neutral — the schedule
    moves exactly the dynamics that compound."""
    fast = make_env("coupled_society")
    slow = make_env("coupled_society", econ_cadence=3)
    _, tf = fast.run_batch(KEY, 6, 400)
    _, ts = slow.run_batch(KEY, 6, 400)
    lf = float(jnp.mean(jax.vmap(fast.metrics["labor_share"])(tf)))
    ls = float(jnp.mean(jax.vmap(slow.metrics["labor_share"])(ts)))
    assert ls > lf + 0.05


def test_coupled_defenses_still_help():
    """Direction check: even coupled, the defended portfolio beats no defenses
    (erosion, not inversion)."""
    def composite(env):
        _, tr = env.run_batch(KEY, n_seeds=6, n_steps=400)
        return float(jnp.mean(jax.vmap(env.metrics["composite_human_share"])(tr)))

    defended = composite(make_env("coupled_society", mechanisms=_defenses()))
    undefended = composite(make_env("coupled_society"))
    assert defended > undefended + 0.05
