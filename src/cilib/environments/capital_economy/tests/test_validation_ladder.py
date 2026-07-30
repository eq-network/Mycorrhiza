"""
Validation ladder for `capital_economy` — WP1's rungs, mechanism not bit-exact.

Rungs (each backs a paper claim; see the WP1 paper, Obsidian vault):
1. Limit equivalence: no arrivals ⇒ exact Leontief stationarity + exact conservation.
2. The knee (Prop 1): capital dies below the pre-registered e*, compounds above.
3. Conservation at every closure r (Prop 2/3) — float32 tolerance.
4. Fault injection: the two prototype bug classes reintroduced deliberately;
   the global probe must catch both (Prop 3's detection claim).
5. Eradication band (Prop 4): the tax moves e* — capital that compounds untaxed
   dies under the scheduled tax at the same e.
6. Ownership fidelity (Cor 1): ω diverts title without changing aggregate capital.
7. Decoupling (Prop Decouple): concentration rises at r=1 and r≈0; output holds
   at r=1 and winds down at r≈0.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from cilib.environments import make_env
from cilib.environments.capital_economy import (
    CapitalEconomyConfig, survival_threshold, money_series,
)
from cilib.environments.capital_economy.state import make_state, technical_matrix
from cilib.lab.analysis.conservation import drift, assert_conserved
from cilib.mechanisms.fiscal import AIRevenueTaxConfig, make_ai_revenue_tax
from cilib.core.schedule import scheduled


CFG = CapitalEconomyConfig()
KEY = jr.PRNGKey(0)


def _run(n_steps=300, mechanisms=(), **over):
    env = make_env("capital_economy", **over)
    if mechanisms:
        from cilib.environments.capital_economy import build_capital_economy
        env = build_capital_economy(mechanisms, **over)
    return env.run(KEY, n_steps=n_steps)


def _sector_va(trace, cfg):
    A = technical_matrix(cfg)
    v = jnp.maximum(1.0 - jnp.sum(A, axis=0), 0.0)
    return v * trace["gross_output"]          # (T, N); sector slots carry va


def test_limit_equivalence_no_arrivals_is_stationary_and_conserved():
    _, tr = _run(first_arrival=10 ** 8)
    y = jnp.sum(tr["gross_output"], axis=-1)
    assert float(jnp.max(jnp.abs(y - y[10]))) < 1e-3          # Leontief fixed point
    cfg = CapitalEconomyConfig(first_arrival=10 ** 8)
    assert drift(money_series(tr, cfg)) < 1e-5                # conservation exact


def test_knee_capital_dies_below_estar_compounds_above():
    # measure v on the people-only economy, evaluate the PAPER's e* expression
    _, tr0 = _run(first_arrival=10 ** 8, n_steps=100)
    cfg = CapitalEconomyConfig()
    va = _sector_va(tr0, cfg)[-1]
    H, S = cfg.n_households, cfg.n_sectors
    v_range = va[H:H + S]
    e_star_lo = survival_threshold(cfg, float(jnp.max(v_range)))
    e_star_hi = survival_threshold(cfg, float(jnp.min(v_range)))

    _, tr_below = _run(efficiency=0.5 * e_star_lo)
    _, tr_above = _run(efficiency=2.0 * e_star_hi)
    k_below = float(jnp.sum(tr_below["capital"][-1]))
    k_above = float(jnp.sum(tr_above["capital"][-1]))
    seeded = cfg.n_owners * cfg.init_capital
    assert k_below < 0.1 * seeded          # dies (whatever the start)
    assert k_above > 2.0 * seeded          # compounds


@pytest.mark.parametrize("r", [1.0, 0.5, 0.0])
def test_conservation_at_every_closure(r):
    cfg = CapitalEconomyConfig(recycle=r)
    _, tr = _run(recycle=r)
    assert_conserved(money_series(tr, cfg), tol=1e-3, label=f"money(r={r})")


def test_fault_injection_probe_catches_both_prototype_bugs():
    # Bug class 1 (savings leak): recompute the invariant WITHOUT the wealth
    # stock — money visibly drains into the untracked hole.
    cfg = CapitalEconomyConfig()
    _, tr = _run()
    leaky = money_series(tr, cfg) - jnp.sum(tr["wealth"], axis=-1)
    assert drift(leaky) > 0.05
    # Bug class 2 (unbacked spending): inject demand from nowhere and check the
    # probe sees creation — simulate by adding a phantom inflow to the series.
    phantom = money_series(tr, cfg) + 0.05 * jnp.arange(tr["wealth"].shape[0])
    assert drift(phantom) > 0.05
    with pytest.raises(AssertionError):
        assert_conserved(phantom, tol=1e-3)


def test_eradication_band_tax_moves_the_threshold():
    # The machines sector has the widest margin, so its owner's knee binds:
    # pick e inside the MACHINES band (e*(0), e*(tau)) — there, untaxed capital
    # compounds while the same e dies under the scheduled tax.
    cfg = CapitalEconomyConfig()
    _, tr0 = _run(first_arrival=10 ** 8, n_steps=100)
    H = cfg.n_households
    v_machines = float(_sector_va(tr0, cfg)[-1][H])             # widest margin
    tau = 0.8
    e_lo = survival_threshold(cfg, v_machines)                  # untaxed knee
    cfg_taxed = CapitalEconomyConfig(reinvest_rate=cfg.reinvest_rate * (1 - tau))
    e_hi = survival_threshold(cfg_taxed, v_machines)            # taxed knee
    e_mid = 0.5 * (e_lo + e_hi)

    tax = scheduled(make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=tau)), onset=0)
    _, tr_untaxed = _run(efficiency=e_mid, n_steps=400)
    _, tr_taxed = _run(efficiency=e_mid, n_steps=400, mechanisms=(tax,))
    k_untaxed = float(jnp.sum(tr_untaxed["capital"][-1]))
    k_taxed = float(jnp.sum(tr_taxed["capital"][-1]))
    seeded = cfg.n_owners * cfg.init_capital
    assert k_untaxed > 1.5 * k_taxed       # the band exists: tax eradicates
    assert k_taxed < 0.5 * seeded          # taxed: dying, well below seed


def test_ownership_two_fund_designs_dividend_split_and_mirror_ratchet():
    # WP1 Prop 4 (as corrected by this implementation): neither fund design is
    # a neutral title split. The DIVIDEND fund converges to pub_share = omega
    # exactly, output ~unchanged, household income up (diversion-as-transfer);
    # the MIRROR fund is a ratchet — common per-unit profit + the multiplier
    # gap drive the public share toward 1 and the economy grows.
    H = CFG.n_households

    def stats(tr):
        k = float(jnp.sum(tr["capital"][-1])) + float(jnp.sum(tr["pub_cap"][-1]))
        share = float(jnp.sum(tr["pub_cap"][-1])) / max(k, 1e-9)
        y = float(jnp.mean(jnp.sum(tr["gross_output"][-50:], axis=-1)))
        hh = float(jnp.mean(jnp.sum(tr["last_reward"][-50:, :H], axis=-1)))
        return k, share, y, hh

    _, tr0 = _run(ownership=0.0)
    _, tr_div = _run(ownership=0.4, pub_mirror=False)
    _, tr_mir = _run(ownership=0.4, pub_mirror=True)
    k0, _, y0, hh0 = stats(tr0)
    k_d, share_d, y_d, hh_d = stats(tr_div)
    k_m, share_m, y_m, _ = stats(tr_mir)

    assert 0.3 < share_d < 0.5                 # dividend: split -> omega
    assert abs(y_d - y0) < 0.05 * y0           # ...output within 5%
    assert hh_d > 1.05 * hh0                   # ...the transfer channel is real
    assert share_m > 0.9                       # mirror: the ratchet
    assert k_m > 1.5 * k_d                     # ...and the economy's stock grows


def test_decoupling_concentration_robust_output_closure_dependent():
    cfg = CapitalEconomyConfig()
    H, S = cfg.n_households, cfg.n_sectors
    O0 = H + S

    def ai_share(tr):
        own = jnp.sum(tr["wealth"][:, O0:] + tr["capital"][:, O0:], axis=-1)
        hum = jnp.sum(tr["wealth"][:, :H], axis=-1) + jnp.sum(tr["pub_cap"][:, H:O0], axis=-1)
        return own / jnp.maximum(own + hum, 1e-8)

    _, tr_ppl = _run(first_arrival=10 ** 8)
    _, tr_r1 = _run(recycle=1.0)
    _, tr_r0 = _run(recycle=0.05)
    y_ppl = float(jnp.mean(jnp.sum(tr_ppl["gross_output"][-50:], axis=-1)))
    y_r1 = float(jnp.mean(jnp.sum(tr_r1["gross_output"][-50:], axis=-1)))
    y_r0 = float(jnp.mean(jnp.sum(tr_r0["gross_output"][-50:], axis=-1)))

    assert float(ai_share(tr_r1)[-1]) > 0.2                    # concentration at r=1
    assert float(ai_share(tr_r0)[-1]) > 0.2                    # ...and at r~0
    assert y_r1 > 0.9 * y_ppl                                  # output holds at r=1
    assert y_r0 < 0.8 * y_ppl                                  # winds down at r~0


def _labor_share_late(tr, cfg):
    H, S = cfg.n_households, cfg.n_sectors
    va = _sector_va(tr, cfg)[:, H:H + S]                       # (T, S)
    ktot = tr["capital"][:, H + S:] + tr["pub_cap"][:, H:H + S]
    a = tr["efficiency"][:, None] * ktot / (tr["efficiency"][:, None] * ktot + 1.0)
    h = jnp.sum((1.0 - a) * va, axis=-1) / jnp.maximum(jnp.sum(va, axis=-1), 1e-8)
    return float(jnp.mean(h[-50:]))


def test_capability_growth_static_exact_and_growth_collapses_share():
    # growth off: e stays exactly at its initial value (limit equivalence)
    _, tr_static = _run()
    assert float(jnp.max(jnp.abs(tr_static["efficiency"] - CFG.efficiency))) == 0.0

    # first-order growth: e rises and the human share falls below the static
    # end state. Second-order (RSI) is a SPEED claim: it crosses any capability
    # level sooner than pure exponential (at matched horizons past the ceiling
    # the shares need not be ordered — both have collapsed). Money is conserved
    # under growth (a moving e only reallocates value-added shares).
    _, tr_exp = _run(growth_rate=0.02)
    _, tr_rsi = _run(growth_rate=0.02, rsi_strength=0.02)
    assert float(tr_exp["efficiency"][-1]) > 2.0 * CFG.efficiency
    level = 8.0 * CFG.efficiency
    t_exp = int(jnp.argmax(tr_exp["efficiency"] > level))
    t_rsi = int(jnp.argmax(tr_rsi["efficiency"] > level))
    assert 0 < t_rsi < t_exp                                   # RSI gets there sooner
    h_static = _labor_share_late(tr_static, CFG)
    assert _labor_share_late(tr_exp, CFG) < h_static
    assert _labor_share_late(tr_rsi, CFG) < h_static
    assert drift(money_series(tr_rsi, CapitalEconomyConfig())) < 1e-3
