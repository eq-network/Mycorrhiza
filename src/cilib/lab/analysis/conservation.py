"""
The global stock-flow conservation probe (WP1 Prop. 3's instrument).

Rationale (WP1 paper §3.1): two prototype bugs — a savings
flow with no return path, and consumption paid from no stock — each produced a
qualitatively wrong economy and were caught only by a *global* conservation
check. Proposition 3 explains why the global probe suffices: if every transform
is locally flow-conservative, their composition is; hence any drift localizes
to a single non-conservative transform by bisection over the pipeline.

Offline math, paper-specific — lives in ``lab.analysis`` per the repo's
where-does-X-go table, not in the environment package.
"""
from __future__ import annotations

import jax.numpy as jnp


def drift(money_series):
    """Max relative deviation of a (T,) money series from its initial value."""
    m = jnp.asarray(money_series)
    return float(jnp.max(jnp.abs(m - m[0])) / jnp.maximum(jnp.abs(m[0]), 1e-8))


def assert_conserved(money_series, tol: float = 1e-3, label: str = "money"):
    """Raise if the series drifts beyond ``tol`` (relative). Returns the drift."""
    d = drift(money_series)
    if d > tol:
        raise AssertionError(
            f"{label} is not conserved: max relative drift {d:.3e} > tol {tol:.1e}. "
            "By WP1 Prop. 3, some transform in the pipeline is non-conservative — "
            "bisect the pipeline to localize it.")
    return d
