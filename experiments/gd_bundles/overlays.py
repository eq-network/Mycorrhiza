"""Analytic overlays shipped in bundle manifests — engine-computed, never a JS
formula (the boundary rule: the page renders artifacts, it does not originate
dynamics). Each entry hands the page a curve in VALUE space; the page's only
job is value->pixel interpolation.

Kinds:
- "boundary": polyline in (x_axis, y_axis) dial space — WP3's pre-registered
  takeover threshold a*(churn), solved from the committed mean-field.
- "band": [lo, hi] interval on one axis — WP1's pre-registered ignition band,
  the e* range from measured people-only value added.
- "curve": one value per axis grid point against a named metric — WP1's
  h* = min(1, e*/e) upper bound on the human sector share.
- "floor": one value per axis grid point against a named metric — WP2's
  analytic floor (1-lam)/(1-lam*s).

ledger_society ships NO overlay: no paper, no committed closed form —
absence is the honest state.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.environments import make_env
from cilib.environments.capital_economy import (
    CapitalEconomyConfig, survival_threshold,
)
from cilib.environments.capital_economy.state import technical_matrix
from experiments.wp3_politics.run import a_star


def _axis(spec, name):
    return next(a for a in spec.axes if a.name == name)


def capital_economy(spec):
    # the WP1 measurement, replicated: run the people-only economy, read the
    # sector value-added row, evaluate the committed e* expression at both
    # band edges (machines-sector v and min v — the pre-registered band)
    cfg = CapitalEconomyConfig(**spec.overrides)
    _, tr0 = make_env("capital_economy", **spec.overrides,
                      first_arrival=10 ** 9).run(jr.PRNGKey(0), 100)
    H, S = cfg.n_households, cfg.n_sectors
    vc = np.asarray(jnp.maximum(
        1.0 - jnp.sum(technical_matrix(cfg), axis=0), 0.0))[H:H + S]
    v_sect = vc * np.asarray(tr0["gross_output"])[-1, H:H + S]
    lo, hi = sorted((survival_threshold(cfg, float(v_sect[0])),
                     survival_threshold(cfg, float(v_sect.min()))))
    eff = _axis(spec, "efficiency").values
    return [
        {"id": "ignition_band", "kind": "band", "axis": "efficiency",
         "label": "pre-registered ignition band (e*)",
         "lo": round(lo, 4), "hi": round(hi, 4)},
        # E5's predicted curve uses the machines edge (h ~ 0.47 bound at
        # e = 0.55 against 0.444 measured) — an upper bound, not a fit
        {"id": "h_star", "kind": "curve", "axis": "efficiency",
         "metric": "human_sector_share",
         "label": "h* = min(1, e*/e) — upper bound on human share",
         "values": [round(min(1.0, lo / e), 4) for e in eff]},
    ]


def influence_exchange(spec):
    # NO floor overlay: WP2's analytic floor (1-lam)/(1-lam*s) binds the
    # paper's attribution share, an instrument computed from the listening
    # matrix — none of this bundle's metrics track it (measured 2026-07-30:
    # human_influence_share is bit-flat across lam). Drawing the floor against
    # a metric that ignores it would be view-side fiction.
    return []


def delegative_polity(spec):
    # dense polyline so the page draws a smooth boundary; a_star solves the
    # committed mean-field from the frozen env config (WP3 run.py). Infinite
    # thresholds (churn-rich rows that never capture) simply leave the plot.
    churns = _axis(spec, "churn").values
    pts = []
    for c in np.linspace(min(churns), max(churns), 19):
        a = a_star(float(c))
        if np.isfinite(a):
            pts.append([round(float(c), 4), round(float(a), 4)])
    return [
        {"id": "a_star", "kind": "boundary",
         "x_axis": "churn", "y_axis": "ai_advantage",
         "label": "pre-registered takeover threshold a*(churn)",
         "points": pts},
    ]


def ledger_society(spec):
    return []


OVERLAYS = {
    "capital_economy": capital_economy,
    "influence_exchange": influence_exchange,
    "delegative_polity": delegative_polity,
    "ledger_society": ledger_society,
}


def overlays_for(spec):
    return OVERLAYS[spec.env](spec)
