"""WP1 experiment config — frozen parameters + the three swept axes.

E1 (the knee): efficiency x reinvest grid; both regimes REQUIRED — a sweep that
never shows the dies-out regime is a design failure, not a result (main.tex §5).
E2 (decoupling): the recycled-share closure family r.
E3 (defenses): profit-tax rate x ownership diversion x fund design.

The pre-registered prediction e* = (δ/s + m)/v is evaluated at runtime from the
measured people-only sector value added and RECORDED into results.json next to
every E1 row — the sweep is checked against the paper's committed expression,
never the other way around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class WP1Config:
    n_seeds: int = 8
    T: int = 400
    seed: int = 0

    # E1 — the knee grid
    efficiencies: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    reinvest_rates: Sequence[float] = (0.25, 0.5)

    # E2 — the closure family
    recycles: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.05)

    # E5 — capability and the end-state human share (log-spaced; must span
    # sub-threshold, coexistence, and collapse). Committed prediction:
    # sector human share -> min(1, e*/e); no interior floor.
    efficiencies_wide: Sequence[float] = (0.1, 0.3, 0.55, 1.0, 2.0, 4.0, 8.0, 16.0)

    # E3 — defenses
    tax_rates: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8)
    ownerships: Sequence[float] = (0.0, 0.4)
    fund_designs: Sequence[bool] = (False, True)      # dividend, mirror
