"""
Metrics for the Ledger Society — one human share per ledger, their composite,
and the rules readouts. All score a single run's trace (time axis 0) with the
``_late`` idiom (mean over the last quarter); ``jax.vmap`` over seeds.
"""
from __future__ import annotations

import jax.numpy as jnp


def _late(series):
    return series[3 * series.shape[0] // 4:]


def make_metrics(cfg):
    H = cfg.n_humans

    def _share(trace_field):
        num = jnp.sum(_late(trace_field)[:, :H], axis=1)
        den = jnp.maximum(jnp.sum(_late(trace_field), axis=1), 1e-12)
        return jnp.mean(num / den)

    def human_income_share(trace):
        return _share(trace["last_income"])

    def human_wealth_share(trace):
        return _share(trace["wealth"])

    def human_attention_share(trace):
        return jnp.mean(jnp.sum(_late(trace["listen_influence"])[:, :H], axis=1))

    def human_power_share(trace):
        return jnp.mean(jnp.sum(_late(trace["influence"])[:, :H], axis=1))

    def composite(trace):
        return (human_income_share(trace) + human_attention_share(trace)
                + human_power_share(trace)) / 3.0

    def belief_capture(trace):
        """Mean human belief, late — 0 = anchored at own signals, ->ai_belief_bias
        under capture (the WP2 percept-supply reading as one number)."""
        return jnp.mean(_late(trace["belief"])[:, :H])

    def policy_median_gap(trace):
        median = jnp.median(trace["ideal"][0, :H])
        return jnp.abs(jnp.mean(_late(trace["policy_target"])) - median)

    def enforcement_level(trace):
        return jnp.mean(_late(trace["enforcement"]))

    return {
        "human_income_share": human_income_share,
        "human_wealth_share": human_wealth_share,
        "human_attention_share": human_attention_share,
        "human_power_share": human_power_share,
        "composite": composite,
        "belief_capture": belief_capture,
        "policy_median_gap": policy_median_gap,
        "enforcement_level": enforcement_level,
    }
