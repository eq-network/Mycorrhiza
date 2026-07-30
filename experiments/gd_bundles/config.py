"""GD-suite sweep bundles — frozen specs (docs/dial-lattice-design.md, L0/L1).

One BundleSpec per playground tab. Grids are IMPORTED from the WP experiment
configs wherever a WP experiment defines them (single source; the papers'
sweeps and the playground's dials cannot drift apart). The ledger_society grid
is new (no experiment yet) — values flagged for Jonas at the Phase C review.

Axis `param_type` follows the house typing: anchored / tuned-for-legibility /
arbitrary-but-swept; `anchor` carries the one-line justification (ordering
claims only) and `paper_ref` the WP section, both shipped in the manifest so a
grid change and its justification cannot drift apart across repos.
`expectations` are committed sign/ordering claims checked by run.py and
RECORDED in results.json — a miss is a reported design failure, never a retune
(WP1 §5 discipline).

v2 (2026-07-30): phase-space-complete grids. Economy moves to WP1 E5's wide
log-efficiency grid and swaps the near-dead reinvest axis (e* moves ~6%
across it) for the E2 closure family r; influence restores the u = 0 row
(WP2's protective switch regime — the earlier drop called it a wasted row,
but flat IS the claim); ledger playback re-centered on the post-fix default
cell. Playback payloads gain the ledger top-target index series + channel
magnitudes that drive the flow scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from experiments.wp1_economy.config import WP1Config
from experiments.wp2_culture.config import WP2Config
from experiments.wp3_politics.config import WP3Config

_WP1, _WP2, _WP3 = WP1Config(), WP2Config(), WP3Config()

N_SEEDS = 8
T = 400
SEED0 = 0
ROUND_DECIMALS = 4
MAX_RUN_BYTES = 400_000
MAX_BUNDLE_BYTES = 12_000_000


@dataclass(frozen=True)
class Axis:
    name: str            # the env-config override field
    label: str           # dial label shown on the page
    values: Tuple[float, ...]
    param_type: str      # anchored | tuned-for-legibility | arbitrary-but-swept
    unit: Optional[str] = None
    anchor: Optional[str] = None     # one-line justification, ordering claims only
    paper_ref: Optional[str] = None  # WP section (None: no paper — say so honestly)


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    direction: str       # up_good | down_good | neutral


@dataclass(frozen=True)
class BundleSpec:
    bundle_id: str
    env: str
    axes: Tuple[Axis, ...]
    metrics: Tuple[Metric, ...]
    whitelist: Tuple[str, ...]          # trace fields shipped in playback cells
    derived: Tuple[str, ...]            # engine-side (T,) reductions (derived.py)
    overrides: dict = field(default_factory=dict)
    playback: object = "all"            # "all" | tuple of index tuples
    T: int = T                          # per-bundle run length (coupled runs longer)
    notes: str = ""


def _subgrid(*index_lists) -> Tuple[Tuple[int, ...], ...]:
    cells: list = [()]
    for idxs in index_lists:
        cells = [c + (i,) for c in cells for i in idxs]
    return tuple(cells)


BUNDLES: Tuple[BundleSpec, ...] = (
    BundleSpec(
        bundle_id="capital-economy-regimes-v2",
        env="capital_economy",
        axes=(
            Axis("efficiency", "AI capability at deployment",
                 tuple(_WP1.efficiencies_wide), "arbitrary-but-swept",
                 anchor=("WP1 E5's log-spaced grid — spans sub-threshold, "
                         "coexistence and collapse; the pre-registered ignition "
                         "band [0.26, 0.70] comes from measured people-only "
                         "value added, and h* = min(1, e*/e) bounds the human "
                         "share above the threshold"),
                 paper_ref="WP1 E1/E5"),
            Axis("recycle", "AI income spent back into the economy",
                 tuple(reversed(_WP1.recycles)), "arbitrary-but-swept",
                 anchor=("WP1 E2's closure family — the committed regime "
                         "feature is an interior output minimum at r = 0.25 "
                         "(partial decoupling starves demand faster than "
                         "hoarding does)"),
                 paper_ref="WP1 E2"),
            Axis("ownership", "public ownership share", (0.0, 0.2, 0.4, 0.6),
                 "arbitrary-but-swept",
                 anchor=("WP1 Prop. 4 — omega diverts title to a public fund, "
                         "output-neutral by construction"),
                 paper_ref="WP1 E3"),
        ),
        metrics=(
            Metric("human_sector_share", "human share of value added", "up_good"),
            # human_income_share re-added for v2: degenerate (1.0) at e <= 0.9,
            # the wide grid reaches e = 16 — drop again if it stays flat there
            Metric("human_income_share", "human income share", "up_good"),
            Metric("ai_wealth_share", "AI wealth share", "down_good"),
            Metric("output_late", "late output", "neutral"),
            Metric("output_peak", "peak output", "neutral"),
            Metric("capital_late", "late AI capital", "neutral"),
            Metric("money_drift", "money conservation drift", "neutral"),
        ),
        whitelist=("capital", "wealth", "efficiency"),
        derived=("human_sector_share", "human_income_share", "ai_wealth_share",
                 "output_total"),
        playback=_subgrid((0, 2, 4, 7), (1, 4), (0, 2)),
        notes=("WP1 regimes grid (efficiencies_wide x recycles imported from "
               "experiments/wp1_economy/config.py) + the ownership (omega) "
               "defense dial. v2 replaces v1's reinvest axis: e* moves only "
               "~6% between s = 0.25 and 0.5, a near-dead dial, while the "
               "closure family r carries E2's decoupling story."),
    ),
    BundleSpec(
        bundle_id="influence-exchange-ampdrift-v2",
        env="influence_exchange",
        axes=(
            Axis("amplification", "AI amplification", tuple(_WP2.amp_grid),
                 "arbitrary-but-swept",
                 anchor=("WP2's threat dial — the human share of graph "
                         "influence falls as amplification rises (the "
                         "committed ordering); the paper's saturation-to-floor "
                         "claim is measured on its attribution instrument, "
                         "upstream of this metric"),
                 paper_ref="WP2 App. B"),
            Axis("update_rate", "attention drift rate", tuple(_WP2.drift_grid),
                 "arbitrary-but-swept",
                 anchor=("WP2's switch — u = 0 freezes the attention graph and "
                         "defeats ANY amplification (the protective regime); "
                         "any u > 0 reaches the same endpoint, u only sets the "
                         "speed"),
                 paper_ref="WP2 App. B"),
            Axis("susceptibility", "susceptibility to the network",
                 tuple(_WP2.floor_lams), "anchored",
                 anchor=("lam, the Friedkin-Johnsen anchor strength (DeGroot "
                         "1974; Friedkin & Johnsen 1990). Moves the belief "
                         "side only: at amp 32 consensus error climbs "
                         "monotonically with lam to total capture at lam = 1. "
                         "The analytic floor (1-lam)/(1-lam*s) binds WP2's "
                         "attribution share, not these graph metrics — "
                         "measured 2026-07-30, v1 lattice"),
                 paper_ref="WP2 §3 / App. B"),
        ),
        metrics=(
            Metric("human_influence_share", "human influence share", "up_good"),
            Metric("influence_gini", "influence Gini", "down_good"),
            Metric("centralization", "centralization", "down_good"),
            Metric("consensus_error", "consensus error", "down_good"),
        ),
        whitelist=("influence",),
        derived=("human_influence_share", "top_influence_share",
                 "opinion_p10", "opinion_p50", "opinion_p90"),
        playback=_subgrid((0, 2, 5), (0, 3, 5), (0, 3)),
        notes=("WP2 dial grids (amp_grid x drift_grid x floor_lams), imported "
               "from experiments/wp2_culture/config.py; susceptibility 1.0 is "
               "the floor-removal corner (capture unbounded). v2 restores "
               "update_rate = 0.0: the flat row IS the paper's switch claim — "
               "frozen attention defeats any amplification — not a wasted row "
               "(reverses the 2026-07-30 drop, same date)."),
    ),
    BundleSpec(
        bundle_id="delegative-polity-knee-v2",
        env="delegative_polity",
        axes=(
            Axis("ai_advantage", "AI delegate advantage", tuple(_WP3.advantages),
                 "arbitrary-but-swept",
                 anchor=("WP3 E1 — takeover once a exceeds the pre-registered "
                         "a*(churn), the saddle-node of the committed "
                         "mean-field (a* = 1 + r/u at gamma = 1); the "
                         "boundary ships as a manifest overlay"),
                 paper_ref="WP3 E1"),
            Axis("churn", "re-delegation churn", tuple(_WP3.churns),
                 "anchored",
                 anchor=("institutionalized uncertainty (Przeworski 1991) — "
                         "churn-rich rows hold against 8x advantage in every "
                         "swept cell"),
                 paper_ref="WP3 §3"),
            Axis("entrenchment_gain", "lock-in strength", tuple(_WP3.lockins),
                 "arbitrary-but-swept",
                 anchor=("WP3 E2 — at gamma > 1 capture persists after the "
                         "advantage is removed in churn-poor rows; 0 is the "
                         "honest region"),
                 paper_ref="WP3 E2"),
        ),
        metrics=(
            Metric("human_power_share", "human power share", "up_good"),
            Metric("influence_gini", "power Gini", "down_good"),
            Metric("centralization", "centralization", "down_good"),
            Metric("top_delegate_share", "top delegate share", "down_good"),
            Metric("policy_median_gap", "policy-median gap", "down_good"),
            Metric("decision_quality", "decision error", "down_good"),
            Metric("enforcement_level", "enforcement", "up_good"),
            Metric("wealth_gini", "citizen wealth Gini", "down_good"),
        ),
        whitelist=("influence", "ideal", "policy_target", "enforcement",
                   "redelegation_friction"),
        derived=("human_power_share", "top_delegate_share"),
        playback=_subgrid((0, 2, 4, 6), (0, 3), (0, 2, 3)),
        notes=("WP3 E1 knee grid (advantage x churn) + the lock-in dial "
               "(entrenchment_gain, E2's axis) — all imported from "
               "experiments/wp3_politics/config.py; lock-in 0 is the honest "
               "region. Axes unchanged from v1 (the grid already crosses "
               "a*(churn) in every row); v2 adds the overlay + metadata."),
    ),
    BundleSpec(
        bundle_id="ledger-society-channels-v2",
        env="ledger_society",
        axes=(
            Axis("reach_per_spend", "money buys reach", (0.0, 1.0, 2.0, 4.0, 8.0),
                 "arbitrary-but-swept",
                 anchor=("no paper — docs/ledger-design.md; values probed "
                         "2026-07-30, 0 seals money->attention")),
            Axis("attention_to_ballots", "attention moves ballots",
                 (0.0, 0.5, 1.0, 2.0, 4.0), "arbitrary-but-swept",
                 anchor=("no paper — docs/ledger-design.md; values probed "
                         "2026-07-30, 0 seals culture->politics")),
            # values resolve the knee against repair_rate=0.02 (probed 2026-07-30:
            # enforcement 1.00/0.87/0.70/0.24/0.01 across these five)
            Axis("regime_rate", "money moves rules",
                 (0.0, 0.005, 0.01, 0.02, 0.04), "arbitrary-but-swept",
                 anchor=("no paper — docs/ledger-design.md; the knee against "
                         "repair_rate = 0.02 probed 2026-07-30, 0 seals "
                         "money->rules")),
        ),
        metrics=(
            Metric("human_income_share", "human income share", "up_good"),
            Metric("human_wealth_share", "human wealth share", "up_good"),
            Metric("human_attention_share", "human attention share", "up_good"),
            Metric("human_power_share", "human power share", "up_good"),
            Metric("composite", "composite human share", "up_good"),
            Metric("belief_capture", "belief capture", "down_good"),
            Metric("policy_median_gap", "policy-median gap", "down_good"),
            Metric("enforcement_level", "enforcement", "up_good"),
        ),
        whitelist=("wealth", "ideal", "efficiency", "policy_target",
                   "enforcement", "top_listen_target", "top_delegate_target"),
        derived=("human_income_share", "human_wealth_share",
                 "human_attention_share", "human_power_share",
                 "belief_mean_human",
                 "bought_reach_human", "bought_reach_ai",
                 "ballot_pull_human", "ballot_pull_ai",
                 "lobby_pressure_human", "lobby_pressure_ai",
                 "net_transfer_human"),
        # 3x3x3 playback subgrid re-centered on the default cell (3,3,2) with
        # the fully sealed twin (0,0,0) and the max corner (4,4,4) reachable
        playback=_subgrid((0, 3, 4), (0, 3, 4), (0, 2, 4)),
        T=800,   # the coupled system runs twice as long — lock-in needs time
        notes=("NEW grid (no prior experiment) — dial values flagged for Jonas "
               "at the review. Cell (0,0,0) is the fully sealed twin; the "
               "regime axis resolves the funded-pressure knee against "
               "institutional repair. v2 ships the top-target index series + "
               "channel-magnitude globals that drive the flow scene."),
    ),
)

# committed sign/ordering expectations: (bundle_id, description, check-key).
# run.py evaluates each against the finished lattice and records pass/fail in
# results.json; any failure exits nonzero (report, revise grid — never retune).
# Thresholds probed 2026-07-30 BEFORE commit (dial-lattice discipline).
EXPECTATIONS = {
    "capital-economy-regimes-v2": [
        ("three regimes along efficiency at r=1.0: sector share > 0.9 at "
         "e=0.1, interior at e=0.55, < 0.1 at e=16", "capital_three_regimes"),
        ("AI wealth share orders with efficiency at r=1.0: small at e=0.1, "
         "large at e=16", "capital_ai_wealth_orders"),
        ("E2's interior output minimum: late output at r=0.25 below both "
         "neighbors at e=0.55", "capital_recycle_minimum"),
    ],
    "influence-exchange-ampdrift-v2": [
        ("amplification captures at the default drift: human share falls by "
         ">0.10 from amp 1 to amp 32 at update_rate 0.08", "influence_amp_captures"),
        ("frozen attention protects: human share > 0.7 at amp 32, u = 0",
         "influence_frozen_protects"),
        ("susceptibility captures beliefs: consensus error at (amp 32, lam 1.0) "
         "exceeds (amp 32, lam 0.7) — lam's only live lane",
         "influence_lambda_belief_capture"),
    ],
    "delegative-polity-knee-v2": [
        ("both regimes: human power share high at (adv 1, churn 0.05), low at "
         "(adv 8, churn 0.02)", "polity_both_regimes"),
        ("churn defends: human power share > 0.6 at (adv 8, churn 0.20)",
         "polity_churn_defends"),
    ],
    "ledger-society-channels-v2": [
        ("coupling costs humans: composite at max dials below the sealed cell "
         "(0,0,0)", "ledger_coupling_costs"),
        ("the regime knee: enforcement at regime_rate 0 exceeds 0.04 by > 0.5 "
         "at default reach/attention", "ledger_regime_knee"),
    ],
}
