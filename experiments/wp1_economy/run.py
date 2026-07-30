"""WP1 sweeps: E1 (knee) · E2 (decoupling) · E3 (defenses). -> results.json

    python -m experiments.wp1_economy.run

Every metric row carries bootstrap CIs over the shared seed batch. The
pre-registered e* band (WP1 paper, Prop. 1) is evaluated from the
measured people-only value added and recorded alongside E1 — a mismatch is a
reported result, not something to retune (main.tex §5's discipline).
"""
from __future__ import annotations

import json
import os

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.capital_economy import (
    CapitalEconomyConfig, survival_threshold, money_series, build_capital_economy,
)
from cilib.environments.capital_economy.state import technical_matrix
from cilib.lab.analysis.bootstrap import bootstrap_ci
from cilib.lab.analysis.conservation import drift
from cilib.mechanisms.fiscal import AIRevenueTaxConfig, make_ai_revenue_tax
from cilib.core.schedule import scheduled

from .config import WP1Config

CFG0 = CapitalEconomyConfig()
H, S = CFG0.n_households, CFG0.n_sectors
O0 = H + S


# --- per-seed reductions (numpy over (seeds, T, N) traces) --------------------

def per_seed_metrics(trs, ccfg):
    cap = np.asarray(trs["capital"])            # (seeds, T, N)
    pub = np.asarray(trs["pub_cap"])
    wea = np.asarray(trs["wealth"])
    out = np.asarray(trs["gross_output"])
    rew = np.asarray(trs["last_reward"])

    own = wea[..., O0:].sum(-1) + cap[..., O0:].sum(-1)          # (seeds, T)
    hum = wea[..., :H].sum(-1) + pub[..., H:O0].sum(-1)
    share = own / np.maximum(own + hum, 1e-8)
    y = out[..., H:O0].sum(-1)
    T4 = 3 * share.shape[1] // 4

    drifts = np.array([
        drift(money_series({k: jnp.asarray(v[i]) for k, v in trs.items()}, ccfg))
        for i in range(cap.shape[0])])

    return {
        "ai_wealth_share": share[:, T4:].mean(1),
        "capital_late": cap[..., O0:].sum(-1)[:, T4:].mean(1),
        "pub_share": (pub[..., H:O0].sum(-1) /
                      np.maximum(pub[..., H:O0].sum(-1) + cap[..., O0:].sum(-1), 1e-8)
                      )[:, T4:].mean(1),
        "output_late": y[:, T4:].mean(1),
        "hh_income_late": rew[..., :H].sum(-1)[:, T4:].mean(1),
        "money_drift": drifts,
    }


def with_cis(per_seed):
    row = {}
    for name, vals in per_seed.items():
        point, lo, hi = bootstrap_ci(np.asarray(vals))
        row[name] = {"point": float(point), "lo": float(lo), "hi": float(hi)}
    return row


def sweep_point(cfg: WP1Config, key, mechanisms=(), **over):
    ccfg = CapitalEconomyConfig(**over)
    env = build_capital_economy(tuple(mechanisms), **over) if mechanisms \
        else make_env("capital_economy", **over)
    _, trs = env.run_batch(key, n_seeds=cfg.n_seeds, n_steps=cfg.T)
    return with_cis(per_seed_metrics(trs, ccfg))


def main():
    cfg = WP1Config()
    key = jr.PRNGKey(cfg.seed)

    # the pre-registered band, from measured people-only value added
    env0 = make_env("capital_economy", first_arrival=10 ** 9)
    _, tr0 = env0.run(jr.PRNGKey(cfg.seed), n_steps=100)
    A = technical_matrix(CFG0)
    v = np.asarray(jnp.maximum(1 - jnp.sum(A, axis=0), 0) * tr0["gross_output"][-1])
    v_sect = v[H:O0]
    results = {"e_star_pred": {
        "v_measured": [float(x) for x in v_sect],
        "bands": {f"s={s}": {"machines": survival_threshold(
                                 CapitalEconomyConfig(reinvest_rate=s), float(v_sect[0])),
                             "min_v": survival_threshold(
                                 CapitalEconomyConfig(reinvest_rate=s), float(v_sect.min()))}
                  for s in cfg.reinvest_rates},
    }}

    e1 = []
    for s in cfg.reinvest_rates:
        for e in cfg.efficiencies:
            row = sweep_point(cfg, key, efficiency=e, reinvest_rate=s)
            e1.append({"efficiency": e, "reinvest_rate": s, **row})
            print(f"E1 e={e} s={s}: K_late={row['capital_late']['point']:.2f}")
    results["E1"] = e1

    e2 = []
    ppl = sweep_point(cfg, key, first_arrival=10 ** 9)
    results["people_only"] = ppl
    for r in cfg.recycles:
        row = sweep_point(cfg, key, recycle=r)
        e2.append({"recycle": r, **row})
        print(f"E2 r={r}: share={row['ai_wealth_share']['point']:.3f} "
              f"Y={row['output_late']['point']:.1f}")
    results["E2"] = e2

    e3 = []
    for tau in cfg.tax_rates:
        mech = (scheduled(make_ai_revenue_tax(AIRevenueTaxConfig(tax_rate=tau)),
                          onset=0),) if tau > 0 else ()
        for om in cfg.ownerships:
            designs = cfg.fund_designs if om > 0 else (False,)
            for mir in designs:
                row = sweep_point(cfg, key, mechanisms=mech, ownership=om, pub_mirror=mir)
                e3.append({"tax_rate": tau, "ownership": om, "pub_mirror": mir, **row})
        print(f"E3 tau={tau} done")
    results["E3"] = e3

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("[saved] results.json")


if __name__ == "__main__":
    main()
