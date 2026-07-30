"""Derived (T,) series per environment — engine-side reductions shipped as
extra `global` fields in playback payloads (contract v1.1 Producers note).

Pure numpy over one run's trace. Each function returns {name: (T,) array} for
exactly the names its BundleSpec declares in `derived`. These mirror the env
metrics' definitions as time series (no late-window mean) so the page's vitals
charts show the same quantities the phase map scores.
"""
from __future__ import annotations

import numpy as np


def _np(trace):
    return {k: np.asarray(v) for k, v in trace.items()}


def capital_economy(trace, cfg):
    from cilib.environments.capital_economy.state import technical_matrix
    t = _np(trace)
    H, S = cfg.n_households, cfg.n_sectors
    O0 = H + S
    own = t["wealth"][:, O0:].sum(1) + t["capital"][:, O0:].sum(1)
    human = t["wealth"][:, :H].sum(1) + t["pub_cap"][:, H:O0].sum(1)
    hh = t["last_reward"][:, :H].sum(1)
    own_w = t["wealth"][:, O0:].sum(1)
    own_flow = np.diff(own_w, prepend=own_w[:1])
    # human_sector_share mirrors the env metric (WP1 E5's h) as a series
    vc = np.maximum(1.0 - np.asarray(technical_matrix(cfg)).sum(0), 0.0)[H:O0]
    va = vc * t["gross_output"][:, H:O0]
    ktot = t["capital"][:, O0:] + t["pub_cap"][:, H:O0]
    e = t["efficiency"][:, None]
    auto = e * ktot / (e * ktot + 1.0)
    return {
        "ai_wealth_share": own / np.maximum(own + human, 1e-8),
        "human_income_share": hh / np.maximum(hh + np.maximum(own_flow, 0.0), 1e-8),
        "human_sector_share": ((1.0 - auto) * va).sum(1)
                              / np.maximum(va.sum(1), 1e-8),
        "output_total": t["gross_output"][:, H:O0].sum(1),
    }


def influence_exchange(trace, cfg):
    t = _np(trace)
    n_c = cfg.n_citizens
    v, x = t["influence"], t["opinion"][:, :n_c]
    return {
        "human_influence_share": v[:, :n_c].sum(1),
        "top_influence_share": v.max(1) / np.maximum(v.sum(1), 1e-12),
        "opinion_p10": np.percentile(x, 10, axis=1),
        "opinion_p50": np.percentile(x, 50, axis=1),
        "opinion_p90": np.percentile(x, 90, axis=1),
    }


def delegative_polity(trace, cfg):
    t = _np(trace)
    v = t["influence"]
    return {
        "human_power_share": v[:, :cfg.n_citizens].sum(1),
        "top_delegate_share": v.max(1) / np.maximum(v.sum(1), 1e-12),
    }


def ledger_society(trace, cfg):
    t = _np(trace)
    H = cfg.n_humans
    N = cfg.n_humans + cfg.n_ai

    def share(field):
        tot = np.maximum(t[field].sum(1), 1e-12)
        return t[field][:, :H].sum(1) / tot

    # channel magnitudes, zone-split. Each mirrors its dynamics.py term and
    # carries the dial multiplier, so a sealed dial (0) makes the series
    # exactly 0 — the page draws these, it never multiplies by a parameter.
    reach = cfg.reach_per_spend * t["broadcast_spend"]          # broadcast_reach
    pull = cfg.attention_to_ballots * N * t["listen_influence"]  # rewire_delegation
    lobby = t["lobby_spend"]                                     # update_regime
    pressure = (cfg.regime_rate * lobby * np.sign(t["net_transfer"])
                / (lobby.sum(1) + cfg.pressure_scale)[:, None])

    return {
        "human_income_share": share("last_income"),
        "human_wealth_share": share("wealth"),
        "human_attention_share": t["listen_influence"][:, :H].sum(1),
        "human_power_share": t["influence"][:, :H].sum(1),
        "belief_mean_human": t["belief"][:, :H].mean(1),
        "bought_reach_human": reach[:, :H].sum(1),
        "bought_reach_ai": reach[:, H:].sum(1),
        "ballot_pull_human": pull[:, :H].sum(1),
        "ballot_pull_ai": pull[:, H:].sum(1),
        "lobby_pressure_human": pressure[:, :H].sum(1),
        "lobby_pressure_ai": pressure[:, H:].sum(1),
        "net_transfer_human": t["net_transfer"][:, :H].sum(1),
    }


DERIVED = {
    "capital_economy": capital_economy,
    "influence_exchange": influence_exchange,
    "delegative_polity": delegative_polity,
    "ledger_society": ledger_society,
}
