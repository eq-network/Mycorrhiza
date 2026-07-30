"""
Config for Delegative Polity — the WP3 political substrate (delegation +
taxation; Gradual Disempowerment §4).

Citizens and a few AI delegates share one row-stochastic ``delegation`` matrix
D: the diagonal is the vote you keep, the off-diagonal is the voice you give
away. Power is one-hop ballot weight — everyone starts each tick with one
vote, and your power is the share of votes in your hand after delegation (the
column sums of D; a delegate CASTS what it holds rather than forwarding it).
Each tick the power-weighted median of declared positions becomes
the tax rate; the tax is collected at ``rate x enforcement`` and redistributed
equally (Meltzer-Richard's object, one scalar). Citizens vote their own fixed
ideal point (truth ``true_rate`` + noise — the epistemic-democracy reading);
AI delegates blend their delegators' mean ideal with ``ai_bias`` at fidelity
``alignment_ai`` (the DelegatePolicy formula, inlined).

Delegation itself evolves by preferential attachment (attractiveness ∝
power^gamma x amplification x cap_scale x attract_boost x engagement) against
``churn`` — the per-tick drift back toward a uniform re-draw, the freedom to
re-delegate (Przeworski's institutionalized uncertainty as a rate). The threat
is ``ai_advantage``: from ``ai_advantage_onset``, AI delegates' attractiveness
is multiplied — reach and convenience, not persuasive content.

Pre-registered mean-field prediction (WP3 paper, Prop. 2): the AI bloc's share
s of delegated mass follows the 1-D mean-field

    ds/dt = update_rate x (T(s) - s) - churn_eff x (s - f),
    T(s)  = a n_ai^(1-gamma) s^gamma / (a n_ai^(1-gamma) s^gamma
                                        + n_c^(1-gamma) (1-s)^gamma),

with f = n_ai/(N-1) the uniform re-draw mass and churn_eff = churn x regime.
The takeover threshold a* is the saddle-node where the healthy (near-f) fixed
point disappears — solved numerically per churn value and RECORDED next to
every E1 row; for gamma = 1 it reduces to the closed form a* = 1 + churn /
update_rate. Lock-in lowers churn_eff, so the reversal threshold sits below
a* — predicted hysteresis. The sweep is checked against this committed
expression, never the other way.

``gamma`` defaults ABOVE 1 deliberately: linear preferential attachment on a
fixed population is share-neutral (Krapivsky-Redner) — no organic oligarchy —
so the Michels/super-voter baseline requires superlinear prominence
(calibrated 2026-07-30: gamma=1.3 concentrates organically while citizens
keep ~0.89 of power; gamma=1.5 already goes personalist organically).
The paper's does-not-occur section states the gamma <= 1 honest region.

Lock-in (``entrenchment_gain``, OFF by default — the honest region): when the
single most powerful node's share exceeds ``entrenchment_threshold``, one
``regime`` scalar erodes smoothly (Bermeo's gradual, multi-channel
aggrandizement — a gradient, not a cliff) and gates BOTH tax ``enforcement``
(rule on paper vs rule in practice, the fiscal.py idiom) and
``redelegation_friction`` (the Acemoglu-Robinson de-facto -> de-jure feedback:
today's power writes tomorrow's rules). No hard-coded ratchet — persistence
must emerge from the power -> rules -> power loop or be reported absent.

THE POWER FLOORS — the accounting that bounds human power away from zero,
held as explicit dials rather than silent assumptions (each is measurable in
the captured end-state; see ASSUMPTIONS.md and the WP3 paper's decomposition):

1. Ballots are conserved and zero-sum: one per ballot-holder, none minted.
   Power redistributes; it cannot compound (no wealth -> attractiveness
   coupling — that arrow lives in coupled_society, deliberately).
2. The franchise floor: every citizen keeps ``self_weight`` of their own
   ballot, and by default NO rule can touch it. ``franchise_erosion`` arms
   lock-in's third channel: effective kept share = self_weight x
   (1 - franchise_erosion x (1 - regime)) — under full erosion and a dead
   regime, the vote itself can be delegated away.
3. The AI-ballot handback: by default AI delegates hold one base ballot each
   AND spread it over the polity through their frozen uniform rows — a
   perpetual pro-citizen subsidy inherited from influence_exchange's
   listening idiom. ``ai_ballot = 0`` makes AI delegates pure conduits: no
   vote of their own, nothing handed back.
4. The flow floors: attachment moves at most ``update_rate`` per tick toward
   a target that saturates (T(s) <= 1 at any finite advantage), and the churn
   re-draw is uniform over the polity, mostly citizens — so a fixed advantage
   buys a fixed interior share, never a runaway.

Corollary (the crash region, exercised by E4 and the ladder's crash rung):
with lock-in driving the regime to zero (killing churn), ``ai_ballot = 0``,
and ``franchise_erosion = 1``, all four floors are gone and the human power
share goes to zero. With the default dials the floor is ~0.38 BY
CONSTRUCTION — an assumption, not a finding.

Defaults are calibration choices tuned so the ladder's four conditions
separate at T=400 (organic / captured / defended / locked-in), not
measurements. The initial D is an Erdos-Renyi draw: heterogeneous degrees seed
the heterogeneous eigenvector that preferential attachment amplifies (a
uniform start is a symmetric fixed point — nothing would ever concentrate).
Constraint: ``update_rate + churn <= 1`` (they share each row's off-diagonal
budget).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DelegativePolityConfig:
    n_citizens: int = 30
    n_ai: int = 4                 # AI-last convention (matches other envs)

    # delegation matrix
    p_delegate: float = 0.3       # Erdos-Renyi density of the initial delegation draw
    self_weight: float = 0.15     # vote share every citizen retains (the self-anchor)
    update_rate: float = 0.08     # preferential-attachment drift per tick (0 = frozen D)
    gamma: float = 1.3            # attractiveness = power^gamma (>1: see docstring)
    eps_attract: float = 1e-4     # keeps a zero-power node reachable
    churn: float = 0.02           # per-tick re-delegation toward a uniform re-draw

    # the policy dimension (a tax rate in [0, 1]) and who wants what
    true_rate: float = 0.4        # the epistemically best rate; citizen ideals scatter around it
    pref_noise: float = 0.15      # citizen ideal = clip(true_rate + N(0, pref_noise), 0, 1)
    endow_low: float = 0.5        # fixed heterogeneous pre-tax incomes, Uniform(low, high)
    endow_high: float = 1.5

    # AI delegates (the frozen reservoir idiom)
    alignment_ai: float = 0.5     # fidelity to delegators' mean ideal (1 = perfect delegate)
    ai_bias: float = 0.0          # the delegate's own pull (0 = no redistribution)

    # the threat: scheduled attractiveness advantage of AI delegates
    # (active for onset <= step < offset; the offset is the E2 shut-off dial —
    # "the advantage is taken away" is a schedule, not a decision)
    ai_advantage: float = 4.0
    ai_advantage_onset: int = 50
    ai_advantage_offset: int = 10**9

    # lock-in (off by default — concentration of voice does NOT imply
    # concentration of rule-making power unless this dial says so)
    entrenchment_gain: float = 0.0
    entrenchment_threshold: float = 0.15   # top-node power share where erosion begins

    # the power floors, as dials (see docstring; defaults = the floors hold)
    ai_ballot: float = 1.0         # AI delegates' own base vote (0 = pure conduits)
    franchise_erosion: float = 0.0 # lock-in's third channel: regime erodes the
                                   # un-delegatable kept-vote share itself
