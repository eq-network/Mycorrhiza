"""WP2 ("Who Fills Your Head?") — experiment knobs only.

Model parameters (susceptibility λ, n_citizens, ...) are read from
``InfluenceExchangeConfig`` at runtime, never duplicated here — one source of
truth, so the paper's Appendix B table cannot drift from the engine.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class WP2Config:
    n_seeds: int = 8
    T: int = 400
    seed: int = 0
    amplifications: Tuple[float, ...] = (1.0, 4.0)  # off / on (paired seeds)
    amp_onset: int = 50
    snapshot_every: int = 10                # exemplar W frames -> snapshots.npz

    # floor sweep: remove the protective assumptions, watch the share follow
    # the analytic floor (1-lam)/(1-lam*s) down to zero.
    floor_lams: Tuple[float, ...] = (0.7, 0.85, 0.95, 1.0)
    floor_self_weights: Tuple[float, ...] = (0.15, 0.0)
    floor_amplification: float = 32.0

    # dial sweeps: every remaining dial across its range, others at defaults,
    # so no picture depends on an unswept arbitrary value.
    amp_grid: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    drift_grid: Tuple[float, ...] = (0.0, 0.02, 0.04, 0.08, 0.16, 0.32)
