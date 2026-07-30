"""WP3 experiment config — frozen parameters + the three swept axes.

E1 (the knee): ai_advantage x churn grid; both regimes REQUIRED — a sweep that
never shows the tracking-democracy regime, or never shows takeover, is a
design failure, not a result (WP1 main.tex §5's discipline).
E2 (lock-in): entrenchment_gain sweep under (a) an advantage shut-off protocol
(recovery vs hysteresis) and (b) the collective-responsiveness instrument
(shift every citizen ideal, from birth and mid-run; outcome = the EFFECTIVE
tax rate, policy x enforcement — the rule in practice).
E3 (defenses): sortition cadence x influence cap x defense onset, under
capture + lock-in.

The pre-registered prediction (delegative_polity/config.py; WP3 paper,
Prop. 2): the AI bloc's delegated share s follows the 1-D mean-field

    ds/dt = u (T(s) - s) - r_eff (s - f),
    T(s)  = a n_ai^(1-g) s^g / (a n_ai^(1-g) s^g + n_c^(1-g) (1-s)^g),

f = n_ai/(N-1), u = update_rate, g = gamma, r_eff = churn x regime. The
takeover threshold a*(r) is the saddle-node where the healthy near-f fixed
point disappears (for g = 1 it reduces to a* = 1 + r/u). ``run.py`` solves
a*(r) from the frozen environment config at runtime and RECORDS it into
results.json next to every E1 row — the sweep is checked against the paper's
committed expression, never the other way around. A mismatch is a reported
result, not something to retune.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WP3Config:
    n_seeds: int = 8
    T: int = 400
    T_reversal: int = 600
    seed: int = 0

    # E1 — the knee grid (advantage x churn phase plane)
    advantages: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    churns: Sequence[float] = (0.02, 0.05, 0.10, 0.20)

    # E2 — lock-in: shut-off protocol + responsiveness instrument
    lockins: Sequence[float] = (0.0, 0.33, 0.67, 1.0)
    shutoff_t: int = 200          # advantage on at 50 (env default), off here
    e2_churns: Sequence[float] = (0.02, 0.05)
                                  # both regimes of the reversal story: at 0.02 the
                                  # mean-field predicts STRUCTURAL bistability (superlinear
                                  # attachment alone sustains capture after shut-off, at any
                                  # lock-in); at 0.05 it predicts recovery — the integrated
                                  # mean-field verdict is recorded next to every row
    cf_delta: float = -0.15       # common citizen ideal shift (downward: clip-safe)
    cf_t_now: int = 267           # the influence-NOW intervention tick (2T/3)
    cf_window: int = 50           # policy window scored after cf_t_now

    # E3 — defenses under capture + lock-in
    e3_lockin: float = 0.67
    sortition_cadences: Sequence[int] = (0, 50, 25, 10)   # 0 = off
    cap_shares: Sequence[float] = (0.0, 0.10, 0.04)       # 0 = off
    defense_onsets: Sequence[int] = (25, 250)             # pre-knee vs post-entrenchment

    # E4 — the power floors, removed cumulatively (captured -> +lock-in kills
    # churn -> +AI ballots removed -> +franchise erosion). The crash check:
    # the final condition must reach human power share < 0.05, or the floors
    # are not actually dials.
    e4_T: int = 800
    e4_lockin: float = 2.0

    # E5 — the assumption sweeps: every load-bearing dial that E1-E4 hold
    # fixed, swept across its range (the paper's assumption map). gamma spans
    # sublinear -> personalist; the AI-objective plane (alpha x bias) includes
    # the aligned column b = true_rate; self_weight spans no-franchise -> a
    # third of every vote kept.
    gammas: Sequence[float] = (0.8, 1.0, 1.15, 1.3, 1.45, 1.6)
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    biases: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8)
    self_weights: Sequence[float] = (0.0, 0.05, 0.15, 0.3)

    # headline block
    headline_lockin: float = 0.67
