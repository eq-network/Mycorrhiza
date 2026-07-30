"""Typed resource manifest — this environment's side of the type contract.

The whitepaper's contract (§ Resources and type contracts) types *transforms*
(reads/writes); this manifest types the *resources* those names refer to, so a
coupling or mechanism can know what it is touching without reading dynamics.py:

  kind       stock (accumulates) | flow (per-tick) | wiring (structure/config
             that happens to live in state) | readout (information copy; the
             underlying quantity is counted elsewhere)
  substance  money | goods | capacity | title | info
  carrier    which node slots (or global) carry meaning
  invariant  role in the money-conservation identity (metrics.money_series);
             None = not part of the conserved stock

One screen, register style. A test asserts this stays exactly in sync with the
state — adding a state field without typing it here is a failure.
"""
from __future__ import annotations

R = dict

RESOURCES = {
    # -- money (the conserved substance; Table 1 of the WP1 paper) -------------
    "last_reward":    R(kind="flow",  substance="money", carrier="household",
                        invariant="pending spending, weight (1 - sigma_s)"),
    "wealth":         R(kind="stock", substance="money", carrier="household+owner",
                        invariant="counted in full (household savings + AI hoard)"),
    "demand_k":       R(kind="flow",  substance="money", carrier="sector",
                        invariant="capital-linked demand in transit (lands next tick)"),
    "demand_h":       R(kind="flow",  substance="money", carrier="sector",
                        invariant=None),  # spent within the tick; covered by last_reward's weight
    "capital_income": R(kind="readout", substance="money", carrier="owner",
                        invariant=None),  # taxable base; the money itself lands in wealth
    "pub_profit":     R(kind="readout", substance="money", carrier="sector",
                        invariant=None),  # fund disposal base; paid out or reinvested same tick
    "upkeep_paid":    R(kind="flow",  substance="money", carrier="global",
                        invariant=None),  # intra-tick relay into demand_k

    # -- goods and productive capacity -----------------------------------------
    "gross_output":   R(kind="flow",  substance="goods", carrier="sector",
                        invariant="inventories in process, 1'Ax (prices = 1)"),
    "capital":        R(kind="stock", substance="capacity", carrier="owner",
                        invariant=None),  # machines, not money; bought via demand_k
    "pub_cap":        R(kind="stock", substance="capacity", carrier="sector",
                        invariant=None),  # the public fund's title-slice of the same
    "efficiency":     R(kind="stock", substance="capacity", carrier="global",
                        invariant=None),  # capability per unit K; grows by the growth law

    # -- wiring (structure that lives in state for shape/scan reasons) ---------
    "spend_pref":     R(kind="wiring", substance="info", carrier="household", invariant=None),
    "spend_weights":  R(kind="wiring", substance="info", carrier="household", invariant=None),
    "active":         R(kind="wiring", substance="info", carrier="all", invariant=None),
    "arrival_tick":   R(kind="wiring", substance="info", carrier="owner", invariant=None),
    "home_sector":    R(kind="wiring", substance="info", carrier="owner", invariant=None),
    "technical":      R(kind="wiring", substance="info", carrier="adj:sector->sector",
                        invariant=None),  # the Leontief recipe layer
    "rng_key":        R(kind="wiring", substance="info", carrier="global", invariant=None),
    "step":           R(kind="wiring", substance="info", carrier="global", invariant=None),
}
