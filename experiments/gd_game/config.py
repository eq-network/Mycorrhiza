"""GD game — frozen deck, windows, three starting towns, committed expectations
(docs/gd-game-design.md, decisions dated 2026-07-31).

Three playable starting conditions over ledger_society (VARIANTS below), all
sharing one deck, one price list and one window schedule; they differ only in
a handful of model dials. Every town runs the money->rules channel at the
probed knee value, so enforcement — the political budget — erodes over the
run. Three intervention windows; at each the player enacts at most one card,
priced against the state the run has actually reached. The tree generator
(run.py) enumerates every affordable branch as a full run from t=0, once per
town; the web page walks the finished tree and computes nothing.

Two currencies, per the 2026-07-31 decision: economic cards are floored on the
human wealth share; political cards debit the enforcement stock in-model
(mechanisms/interventions.py). There is no invented pooled currency.

Every price, rate and town dial here is typed tuned-for-legibility with a
probe date — none is anchored, and the manifest ships that typing. Committed
EXPECTATIONS are sign/ordering claims about the model; a failure is reported
and the design revised, never retuned. The tray property (later windows afford
less) is a declared DESIGN gate: values may be tuned to realize it, and carry
that type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

N_SEEDS = 8
SEED0 = 0
T = 400
WINDOWS = (100, 200, 300)      # tuned-for-legibility; probed 2026-07-31 against
                               # the enforcement/wealth erosion curves (run.py
                               # prints the wait-path resources at each window)
ROUND_DECIMALS = 4
MAX_RUN_BYTES = 400_000
MAX_TREE_BYTES = 40_000_000    # ~120 paths x <=400 KB; the page fetches only
                               # the branches a player actually walks

# card effect parameters (all tuned-for-legibility, probed 2026-07-31)
LEVY_RATE = 0.10
CAMPAIGN_SHIFT = 0.10
REPAIR_SPEND_RATE = 0.02
REPAIR_EFFICIENCY = 0.5
REACH_CUT = 0.9
SORTITION_SHARE = 0.3
SORTITION_CADENCE = 25


@dataclass(frozen=True)
class Card:
    card_id: int        # 1..5; 0 is reserved for "wait"
    key: str
    label: str
    currency: str       # "enforcement" (in-model debit) | "wealth" (share floor)
    price: float        # enforcement units, or the human-wealth-share floor
    param_type: str
    anchor: str


# Prices re-probed 2026-07-31 (second pass, same day) ON HARBOR, and shared
# by all three towns: the forced window-0 repair floats harbor's wait-path
# enforcement arc to 0.727/0.516/0.402, so the political prices moved up to
# restore the designed tray arc — full at window 1, two cards at window 2,
# EMPTY at window 3 (the design gate). The debit IS the price, so costlier
# cards also drain more capacity in-model. The card lines below describe
# harbor's tray; the other towns reach their own trays on these same prices.
CARDS: Tuple[Card, ...] = (
    Card(1, "ai_wealth_levy", "Tax AI wealth", "enforcement", 0.45,
         "tuned-for-legibility",
         "cheapest political card — affordable at windows 1 and 2 on the "
         "wait path, gone by window 3 (re-probed 2026-07-31)"),
    Card(2, "sortition", "Ballot lottery", "enforcement", 0.65,
         "tuned-for-legibility",
         "the constitutional fight — most expensive card; window 1 only on "
         "the wait path (re-probed 2026-07-31)"),
    Card(3, "influence_cap", "Cap bought reach", "enforcement", 0.55,
         "tuned-for-legibility",
         "window 1 only on the wait path (re-probed 2026-07-31)"),
    Card(4, "attention_campaign", "Fund civic broadcasting", "wealth", 0.30,
         "tuned-for-legibility",
         "floor on the human wealth share — a poor human bloc cannot divert "
         "consumption to reach; holds through window 2, gone by window 3 "
         "(re-probed 2026-07-31)"),
    Card(5, "fund_repair", "Fund institutional repair", "wealth", 0.20,
         "tuned-for-legibility",
         "the forced window-0 card (2026-07-31): baked into every run at the "
         "forced tick, no enforcement debit, not choosable at real windows"),
)

CARD_BY_ID = {c.card_id: c for c in CARDS}

# cards considered and cut, with reasons — shipped in the manifest so the
# debrief can show the deck's omissions (the review session's cut-cards ask)
CUT_CARDS = (
    {"key": "channel_seal", "label": "Seal a coupling channel outright",
     "reason": "subsumed — the influence cap at reach_cut 1.0 is the seal; "
               "shipping both would be one knob twice"},
    {"key": "disclosure", "label": "Publish the ledger (make drift visible "
     "early at private cost)",
     "reason": "not modeled — needs an information asymmetry the substrate "
               "does not carry yet; strongest future mechanic on record"},
    {"key": "reserve", "label": "Hold a reserve (spend nothing, keep capacity)",
     "reason": "cannot score inside a single run — the wait action is the "
               "closest the one-run frame can honestly offer"},
)


@dataclass(frozen=True)
class Metric:
    id: str             # env metric id (ledger_society/metrics.py)
    label: str
    direction: str      # up_good | down_good | neutral
    kind: str           # O0 closed vocabulary: share | index | level | rate | flag
    version: int = 1


METRICS: Tuple[Metric, ...] = (
    Metric("human_wealth_share", "human wealth share", "up_good", "share"),
    Metric("human_attention_share", "human attention share", "up_good", "share"),
    Metric("human_power_share", "human power share", "up_good", "share"),
    Metric("composite", "composite human share", "up_good", "index"),
    Metric("enforcement_level", "enforcement", "up_good", "level"),
)


NOTES = ("One deck, three windows, five cards, at most one card per window "
         "and no re-enactment. Branch = full run from t=0 with a fixed plan; "
         "affordability judged on the median across the seed batch at the "
         "tick before each window.")


@dataclass(frozen=True)
class GameSpec:
    # the two fields a town sets; everything below is shared across towns
    tree_id: str
    # ledger_society overrides — the losing scenario in every town: default
    # coupled dials with money->rules at the probed knee (regime_rate 0.02 ->
    # late enforcement ~0.24 against repair 0.02, probe of 2026-07-30), so the
    # political budget visibly erodes in-run
    overrides: dict
    env: str = "ledger_society"
    T: int = T
    windows: Tuple[int, ...] = WINDOWS
    # window 0, forced and free (design consult 2026-07-31): fund repair is
    # baked into EVERY run at this tick as a tutorialized first move — it
    # carries no enforcement debit, so "free" is true in-model. The card
    # leaves the playable deck; real windows offer the other four + wait.
    forced_tick: int = 75
    forced_card_id: int = 5
    metrics: Tuple[Metric, ...] = METRICS
    whitelist: Tuple[str, ...] = (
        "wealth", "ideal", "efficiency", "policy_target", "enforcement",
        "intervention_spend", "top_listen_target", "top_delegate_target",
        "listen_influence", "influence")
    derived: Tuple[str, ...] = (
        "human_income_share", "human_wealth_share", "human_attention_share",
        "human_power_share", "belief_mean_human",
        "bought_reach_human", "bought_reach_ai",
        "ballot_pull_human", "ballot_pull_ai",
        "lobby_pressure_human", "lobby_pressure_ai", "net_transfer_human")
    notes: str = NOTES


@dataclass(frozen=True)
class Variant:
    """One starting town: a tree id plus the model dials that make it that
    town. Deck, prices, windows, forced window, whitelist, derived series, T
    and the seed batch are shared (GameSpec above) — a town is dials only."""
    variant_id: str      # harbor | boomtown | commune
    tree_id: str
    overrides: dict      # ledger_society config overrides
    label: str


# The three starting towns (2026-07-31 design decision: Civ-style starting
# conditions). Every dial below is typed tuned-for-legibility and probed
# 2026-07-31 against the wait-path tray arc run.py prints — none is anchored,
# and each town's tree.json ships its own overrides next to the shared prices.
# Harbor is the previously probed scenario, unchanged. The other two vary
# dials that already exist in ledger_society's config; no price, card or model
# change was made for them.
VARIANTS: Tuple[Variant, ...] = (
    Variant("harbor", "ledger-society-game-harbor-v1",
            {"regime_rate": 0.02},
            "Harbor: the probed scenario unchanged — default hoard, default "
            "institutional repair"),
    # boomtown probe (wait path, 8 seeds, 2026-07-31): enforcement
    # 0.663/0.095/0.000, human wealth share 0.508/0.080/0.002 — tray 3/0/0
    # political cards. Passes tray_shrinks, and note what it means: the town
    # has ONE live decision, at window 1; after that the run is over in
    # everything but name. The starting hoard does not protect it — halving
    # repair against the same regime_rate is what decides the run.
    Variant("boomtown", "ledger-society-game-boomtown-v1",
            {"regime_rate": 0.02, "init_wealth": 2.0, "repair_rate": 0.01},
            "Boomtown: rich households, weak institutions — twice the "
            "starting hoard, half the institutional self-repair"),
    # commune repair_rate probe (wait path, 8 seeds, 2026-07-31): 0.04 as
    # first drafted held enforcement flat at 0.820/0.778/0.777 (tray 3/3/3)
    # and MISSED tray_shrinks; 0.035 (0.804/0.744/0.740) and 0.03
    # (0.785/0.699/0.686) missed it too, both 3/3/3. 0.025 realizes the gate:
    # 0.763/0.638/0.617, tray 3/2/2. Revised here under the design gate's
    # declared licence — this town's dial only; the shared deck and prices,
    # and harbor, were not touched.
    Variant("commune", "ledger-society-game-commune-v1",
            {"regime_rate": 0.02, "init_wealth": 0.6, "repair_rate": 0.025,
             "churn": 0.08},
            "Commune: poor households, strong civics — a smaller hoard "
            "against above-default repair and freer re-delegation"),
)

VARIANT_BY_ID = {v.variant_id: v for v in VARIANTS}


def variant_spec(v: Variant) -> GameSpec:
    """The shared spec carrying one town's tree id, dials and label."""
    return GameSpec(tree_id=v.tree_id, overrides=dict(v.overrides),
                    notes=f"{NOTES} Starting town — {v.label}.")


# committed sign/ordering expectations over tree paths (path = card id per
# window, 0 = wait), keyed by variant_id. Evaluated by run.py on the full
# tree, recorded in results.json; any failure exits nonzero — report and
# revise, never retune.
#
# They bind HARBOR ONLY (2026-07-31). Boomtown and commune are held to the
# tray_shrinks design gate alone: their dials were chosen for playability and
# probed on the same day they were written, so committing the same orderings
# for them would be asserting claims nobody has probed on those dials. Their
# path values are exported and can be read; they are not claimed here.
EXPECTATIONS = {
    "harbor": [
        ("the levy at window 1 beats waiting on the composite share",
         "levy_early_beats_wait"),
        ("the levy helps more at window 1 than at window 2, the latest "
         "wait-path window where it is still affordable (window 3's tray is "
         "empty, probed 2026-07-31)", "levy_early_beats_late"),
        # repair_defends_rules retired 2026-07-31: fund repair is the forced
        # window-0 card in every run, no longer a choosable defense to compare
        ("the reach cap defends the human attention share",
         "cap_defends_attention"),
    ],
}

# the declared DESIGN gate, checked for EVERY town (prices and town dials
# tuned to realize it, typed above): the wait path's tray shrinks — window 3
# affords strictly fewer political cards than window 1, and at least one card
# is always affordable at window 1. Shared prices are frozen by harbor; a town
# that misses this gate is fixed by revising that town's dials, never the deck.
DESIGN_GATES = [
    ("waiting shrinks the tray: fewer political cards affordable at window 3 "
     "than window 1 on the wait path", "tray_shrinks"),
]
