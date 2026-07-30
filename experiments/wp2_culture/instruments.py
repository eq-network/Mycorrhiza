"""Offline instruments for WP2 — pure numpy over saved listening matrices.

The engine model (``influence_exchange``) is untouched; everything here is
presentation-layer math on its snapshots.

* ``attribution`` — the paper's one new object. For a frozen attention matrix
  W and anchor strength 1-λ, each citizen's settled percept is a provable
  convex mix of sources (Friedkin-Johnsen fixed point): solve
  ``(I - λ W_CC) B = [(1-λ) I | λ W_CA]``. Rows sum to 1 exactly because W is
  row-stochastic (asserted at runtime), so ``B[i, j]`` reads "the share of
  citizen i's settled percept supplied by source j".
* ``mode_alignments`` — the resonant-mode picture: which Laplacian mode of the
  symmetrized attention graph best matches the human/AI split. Mirrors the
  phi formula of ``cilib.metrics.families.spectral.fiedler_partition_alignment``
  generalized to every mode, reimplemented here so the library stays untouched.
* ``spring_layout`` — a small deterministic Fruchterman-Reingold so the
  figures need no networkx dependency.

Type contracts (coupling grammar: docs/ledger-design.md, docs/gd-suite-v0.1.md,
whitepaper "Resources and type contracts"). W is the conserved attention
LEDGER: float, row-stochastic, any leading batch axes over a trailing (N, N).
Everything here is a pure reduction of W — nothing writes state. B, power,
and the human share are CANDIDATE PORTS: computed offline today, promoted to
in-engine reductions the day another domain reads them (the engine's own
port is `influence`).

    attribution(W, lam, n_c) -> B: (..., n_c, N); rows sum to 1 (ASSERTED)
    ai_share(B, n_c)         -> (..., n_c); each value in [0, lam]
    power(B)                 -> (..., N); nonnegative, sums to n_c
    anchor_floor(lam, s)     -> scalar in [0, 1]; 0 exactly at lam=1
    mode_alignments(W, t)    -> (vals ascending, vecs orthonormal, phis in [0, 1])
    spring_layout(W, seed)   -> (N, 2); deterministic for a given seed
"""
from __future__ import annotations

import numpy as np


def attribution(W, lam: float, n_c: int):
    """Percept-attribution matrix B, shape ``(..., n_c, N)``.

    Columns ``0..n_c-1`` are the citizens' own anchors, ``n_c..`` the pinned AI
    voices. Batched over any leading axes of ``W``. Raises if a row fails to
    sum to 1 (the row-stochasticity of W makes this an identity, not a check
    of the data — a failure means the solve itself went wrong).
    """
    W = np.asarray(W, dtype=np.float64)
    Wcc = W[..., :n_c, :n_c]
    Wca = W[..., :n_c, n_c:]
    A = np.broadcast_to(np.eye(n_c), Wcc.shape) - lam * Wcc
    eye = np.broadcast_to(np.eye(n_c), Wcc.shape)
    rhs = np.concatenate([(1.0 - lam) * eye, lam * Wca], axis=-1)
    B = np.linalg.solve(A, rhs)
    err = float(np.max(np.abs(B.sum(-1) - 1.0)))
    assert err < 1e-6, f"attribution rows must sum to 1 (max err {err:.2e})"
    return B


def anchor_floor(lam: float, self_weight: float) -> float:
    """Analytic lower bound on the human share: (1-lam)/(1-lam*self_weight).

    Own-anchor share is (1-lam)*M_ii, and M_ii >= sum_k (lam*self_weight)^k
    because the depth-k pure self-loop path contributes exactly that term.
    At lam=1 the floor is zero: nothing in the dynamics resists full capture.
    """
    return (1.0 - lam) / (1.0 - lam * self_weight)


def ai_share(B, n_c: int):
    """Per-citizen share of percept supplied by AI voices, shape ``(..., n_c)``."""
    return B[..., n_c:].sum(-1)


def power(B):
    """Per-source share of everyone's percepts (column sums), shape ``(..., N)``."""
    return B.sum(-2)


def _phi(split, types) -> float:
    """|Matthews correlation| between two binary partitions (sign-invariant)."""
    a = np.asarray(split, dtype=bool)
    b = np.asarray(types, dtype=bool)
    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return abs(n11 * n00 - n10 * n01) / max(denom, 1e-8)


def mode_alignments(W, types):
    """Laplacian modes of the symmetrized attention graph vs the human/AI split.

    Returns ``(vals, vecs, phis)`` with ``phis[k-1]`` = alignment of mode k
    (mode 0, the constant vector, is skipped). Best mode = ``argmax(phis) + 1``.
    """
    W = np.asarray(W, dtype=np.float64)
    Ws = (W + W.T) / 2.0
    np.fill_diagonal(Ws, 0.0)
    L = np.diag(Ws.sum(1)) - Ws
    vals, vecs = np.linalg.eigh(L)
    phis = np.array([_phi(vecs[:, k] >= 0, types) for k in range(1, W.shape[0])])
    return vals, vecs, phis


def spring_layout(W, seed: int = 0, iters: int = 300):
    """Small deterministic Fruchterman-Reingold; returns ``(N, 2)`` positions."""
    W = np.asarray(W, dtype=np.float64)
    A = (W + W.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A = A / max(A.max(), 1e-12)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))
    k = np.sqrt(4.0 / n)  # ideal spacing for n nodes in a [-1, 1]^2 box
    for it in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-6)
        unit = delta / dist[..., None]
        rep = (k * k / dist)[..., None] * unit
        att = (A * dist / k)[..., None] * unit
        disp = (rep - att).sum(axis=1)
        norm = np.maximum(np.linalg.norm(disp, axis=-1, keepdims=True), 1e-9)
        step = 0.1 * (1.0 - it / iters) + 1e-3
        pos = pos + step * disp / norm
        pos = pos - pos.mean(0)
    return pos
