"""
The shared attachment kernel — conserved-share preferential reallocation.

One mathematical object appears in every GD-suite domain that carries a
row-stochastic ledger (attention in ``influence_exchange``, ballots in
``delegative_polity``, both in ``ledger_society``): each row drifts a bounded
step toward an attractiveness distribution, optionally leaks toward a uniform
re-draw (churn — Przeworski's institutionalized uncertainty as a rate), keeps
a fixed diagonal floor (the share you never give away), and never mints or
destroys share. ``docs/gd-suite-v0.1.md`` §2.4 names this the suite's shared
kernel; this module is its single home for new environments.

The two older environments keep their own inlined copies for now — replacing
them is a behavior-preserving refactor gated on bit-identity (CLAUDE.md,
"Verifying a change"), deliberately not done in the same session this module
landed.

Sibling of ``networks.py``: pure array math shared across environments, not a
``@transform`` (each environment wraps it with its own reads/writes).
"""
from __future__ import annotations

import jax.numpy as jnp


def preferential_reallocation(W, attract, update_rate, self_weight,
                              churn=0.0, frozen_rows=None):
    """One conserved-share attachment step on a row-stochastic matrix.

    Each unfrozen row keeps ``self_weight`` on the diagonal and mixes its
    off-diagonal mass: ``(1 - update_rate - churn)`` stays where it was,
    ``update_rate`` moves toward ``attract`` (normalized off-diagonal, never
    toward self), ``churn`` moves toward the uniform re-draw. Guards preserved
    from the originals: a row facing an all-silent world (zero attract mass)
    keeps its current listening; a pure self-voter (zero off-diagonal mass) is
    a fixed point. Rows sum to 1 exactly before and after — the kernel is pure
    reallocation, which is what makes the fields it moves ledgers.

    ``frozen_rows``: optional (N,) bool — rows returned unchanged (the
    reservoir idiom: AI rows are delegated/listened TO; their own row never
    drifts).
    """
    N = W.shape[0]
    eye = jnp.eye(N, dtype=W.dtype)

    target = attract[None, :] * (1.0 - eye)                 # never toward self
    t_mass = jnp.sum(target, axis=1, keepdims=True)
    target = target / jnp.maximum(t_mass, 1e-12)
    uniform = (1.0 - eye) / (N - 1)

    offdiag = W * (1.0 - eye)
    off_mass = jnp.sum(offdiag, axis=1, keepdims=True)
    off_norm = offdiag / jnp.maximum(off_mass, 1e-12)

    drift = ((1.0 - update_rate - churn) * off_norm
             + update_rate * target + churn * uniform)
    mixed = jnp.where(t_mass > 1e-9, drift, off_norm)       # all-silent guard
    W_new = self_weight * eye + (1.0 - self_weight) * mixed
    W_new = jnp.where(off_mass > 1e-9, W_new, W)            # self-voter fixed point

    if frozen_rows is not None:
        W_new = jnp.where(frozen_rows[:, None], W, W_new)
    return W_new
