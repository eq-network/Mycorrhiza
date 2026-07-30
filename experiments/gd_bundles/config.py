"""GD-suite sweep bundles — frozen specs (docs/dial-lattice-design.md, L0/L1).

One BundleSpec per playground tab. Grids are IMPORTED from the WP experiment
configs wherever a WP experiment defines them (single source; the papers'
sweeps and the playground's dials cannot drift apart). The ledger_society grid
is new (no experiment yet) — values flagged for Jonas at the Phase C review.

Axis `param_type` follows the house typing: anchored / tuned-for-legibility /
arbitrary-but-swept. `expectations` are committed sign/ordering claims checked
by run.py and RECORDED in results.json — a miss is a reported design failure,
never a retune (WP1 §5 discipline).
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
        bundle_id="capital-economy-knee-v1",
        env="capital_economy",
        axes=(
            Axis("efficiency", "AI capability at deployment", tuple(_WP1.efficiencies),
                 "arbitrary-but-swept"),
            Axis("reinvest_rate", "profit reinvestment rate", tuple(_WP1.reinvest_rates),
                 "tuned-for-legibility"),
            Axis("ownership", "public ownership share", (0.0, 0.2, 0.4, 0.6),
                 "arbitrary-but-swept"),
        ),
        metrics=(
            # human_income_share dropped: degenerate at this window (== 1.0 in
            # every cell, 2026-07-30 audit) — a dead lane is worse than no lane
            Metric("ai_wealth_share", "AI wealth share", "down_good"),
            Metric("output_late", "late output", "neutral"),
            Metric("output_peak", "peak output", "neutral"),
            Metric("capital_late", "late AI capital", "neutral"),
            Metric("money_drift", "money conservation drift", "neutral"),
        ),
        whitelist=("capital", "wealth", "efficiency"),
        derived=("human_income_share", "ai_wealth_share", "output_total"),
        playback=_subgrid((0, 2, 4, 6, 8), (0, 1), (0, 2)),
        notes=("WP1 E1 knee grid (efficiency x reinvest imported from "
               "experiments/wp1_economy/config.py) + the ownership (omega) defense "
               "dial (WP1 Prop. 4; title diversion, output-neutral by construction)."),
    ),
    BundleSpec(
        bundle_id="influence-exchange-ampdrift-v1",
        env="influence_exchange",
        axes=(
            Axis("amplification", "AI amplification", tuple(_WP2.amp_grid),
                 "arbitrary-but-swept"),
            Axis("update_rate", "attention drift rate", tuple(_WP2.drift_grid[1:]),
                 "arbitrary-but-swept"),
            Axis("susceptibility", "susceptibility to the network",
                 tuple(_WP2.floor_lams), "anchored"),
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
        playback=_subgrid((0, 1, 3, 5), (0, 2, 4), (0, 3)),
        notes=("WP2 dial grids (amp_grid x drift_grid x floor_lams), imported from "
               "experiments/wp2_culture/config.py; drift = update_rate override "
               "(wp2 run.py convention); susceptibility 1.0 is the floor-removal "
               "corner (capture unbounded). update_rate=0.0 dropped: a frozen "
               "attention graph makes every other dial dead, wasting a lattice row."),
    ),
    BundleSpec(
        bundle_id="delegative-polity-knee-v1",
        env="delegative_polity",
        axes=(
            Axis("ai_advantage", "AI delegate advantage", tuple(_WP3.advantages),
                 "arbitrary-but-swept"),
            Axis("churn", "re-delegation churn", tuple(_WP3.churns),
                 "anchored"),
            Axis("entrenchment_gain", "lock-in strength", tuple(_WP3.lockins),
                 "arbitrary-but-swept"),
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
               "experiments/wp3_politics/config.py; lock-in 0 is the honest region."),
    ),
    BundleSpec(
        bundle_id="ledger-society-channels-v1",
        env="ledger_society",
        axes=(
            Axis("reach_per_spend", "money buys reach", (0.0, 1.0, 2.0, 4.0, 8.0),
                 "arbitrary-but-swept"),
            Axis("attention_to_ballots", "attention moves ballots",
                 (0.0, 0.5, 1.0, 2.0, 4.0), "arbitrary-but-swept"),
            # values resolve the knee against repair_rate=0.02 (probed 2026-07-30:
            # enforcement 1.00/0.87/0.70/0.24/0.01 across these five)
            Axis("regime_rate", "money moves rules",
                 (0.0, 0.005, 0.01, 0.02, 0.04), "arbitrary-but-swept"),
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
        whitelist=("wealth", "ideal", "efficiency", "policy_target", "enforcement"),
        derived=("human_income_share", "human_wealth_share",
                 "human_attention_share", "human_power_share", "belief_mean_human"),
        # 3x3x3 playback subgrid including every sealed 0-plane cell within it
        playback=_subgrid((0, 2, 4), (0, 2, 4), (0, 2, 4)),
        T=800,   # the coupled system runs twice as long — lock-in needs time
        notes=("NEW grid (no prior experiment) — dial values flagged for Jonas at "
               "the review. Cell (0,0,0) is the fully sealed twin; the regime axis "
               "resolves the funded-pressure knee against institutional repair."),
    ),
)

# committed sign/ordering expectations: (bundle_id, description, check-key).
# run.py evaluates each against the finished lattice and records pass/fail in
# results.json; any failure exits nonzero (report, revise grid — never retune).
EXPECTATIONS = {
    "capital-economy-knee-v1": [
        ("both regimes present at reinvest 0.5: AI wealth share small at e=0.1, "
         "large at e=0.9", "capital_both_regimes"),
    ],
    "influence-exchange-ampdrift-v1": [
        ("amplification captures at the default drift: human share falls by "
         ">0.10 from amp 1 to amp 32 at update_rate 0.08", "influence_amp_captures"),
    ],
    "delegative-polity-knee-v1": [
        ("both regimes: human power share high at (adv 1, churn 0.05), low at "
         "(adv 8, churn 0.02)", "polity_both_regimes"),
    ],
    "ledger-society-channels-v1": [
        ("coupling costs humans: composite at max dials below the sealed cell "
         "(0,0,0)", "ledger_coupling_costs"),
    ],
}
