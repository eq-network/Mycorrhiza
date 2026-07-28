"""
Config for the Value Contagion substrate — cultural register entry C2
(docs/cultural-register-design.md §11).

Culture as *something you catch*: agents hold one cultural variant, tracked only
by origin (human- or AI-origin), and adopt neighbors' variants contagion-style
on a fixed friendship network. The register's two dials are both plain
parameters (masking — the frozen GameSpec boundary is untouched):

- **axis S (separation):** ``ai_homophily``, passed to
  ``networks.typed_homophily`` — do AI agents mix with humans or talk mostly
  to each other?
- **axis P (persuasive advantage):** ``p_advantage`` — AI-origin variants are
  that many times more transmissible, whoever currently carries them.

Calibration sketch (defaults below; mean-field at degree 6, 32 humans + 8 AI):

    conversion pressure on a fresh human ~ 1 − (1 − beta·P)^1.2      per tick
    reversion pressure on a convert      ~ 1 − (1−recovery)(1−beta)^m_h ≈ 0.27
    P=1: conversion ≈ 0.04 ≪ reversion            -> pluralism (share stays high)
    P=6: per-contact rate 0.18, branching ratio ≈ 5.9·0.18/0.27 > 1  -> sweep
    high S: ~4 cross-type edges in total — a trickle (P=1) or a fuse (P=6)

``recovery`` is load-bearing: AI nodes are a frozen reservoir of AI-origin
culture, so without native reversion the only mean-field fixed point is
"everyone AI-cultured" and the pluralism corner cannot exist. Defaults are
calibration choices tuned so the (S, P) corners separate at 200 steps — not
measurements.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ValueContagionConfig:
    n_agents: int = 40
    n_ai: int = 8                 # AI-last convention (matches typed_homophily)

    # network (axis S)
    mean_degree: float = 6.0
    ai_homophily: float = 0.05    # 0 = fully mixed; ≤ ~0.9 (1.0 disconnects)

    # contagion (axis P)
    beta: float = 0.03            # base per-source per-tick adoption probability
    p_advantage: float = 1.0      # transmissibility multiplier for AI-origin variants
    recovery: float = 0.15        # humans holding AI-origin culture revert natively
    k_threshold: int = 1          # complex-contagion gate: min exposures to adopt

    # representation, not behaviour: store `friendship` as a sparse BCOO so
    # aggregation costs edges rather than N². Same generator, same key, same
    # graph — tests/test_sparse_equivalence.py pins the trajectories together.
    # Off by default; worth turning on around n_agents >~ 1000, where the dense
    # (N, N) matmul dominates. `sparse_nse` overrides the derived bound (see
    # networks.sparse_nse_bound) for configs whose degree distribution is
    # unusual enough that the default headroom would truncate.
    sparse_friendship: bool = False
    sparse_nse: int | None = None
