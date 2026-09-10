"""Branch construction — the single source for "choices -> environment".

A branch is a full run from t=0 with a fixed intervention plan (determinism
with a shared seed makes prefixes identical, so no state-resume machinery
exists anywhere). Both run.py and export.py build environments through this
module; if they drifted apart, the exported playback would not be the run the
tree scored.

Choices are a tuple of card ids, one per window, 0 = wait. Card mapping
(config.CARDS): the influence cap is a channel property and enters as config
overrides; sortition is the existing political mechanism on a schedule; levy,
campaign and fund-repair ride the composed interventions transform, which is
always present (an empty plan is a bit-exact identity — tested).
"""
from __future__ import annotations

from typing import Tuple

from cilib.core.schedule import scheduled
from cilib.mechanisms import (
    InterventionPlanConfig, make_interventions,
    SortitionConfig, make_sortition,
)

from .config import (
    CAMPAIGN_SHIFT, CARD_BY_ID, LEVY_RATE, REACH_CUT, REPAIR_EFFICIENCY,
    REPAIR_SPEND_RATE, SORTITION_CADENCE, SORTITION_SHARE, GameSpec,
)


def branch_kwargs(spec: GameSpec, choices: Tuple[int, ...]):
    """``(overrides, mechanisms)`` for ``make_env(spec.env, mechanisms=..., **overrides)``."""
    assert len(choices) == len(spec.windows)
    seen = [c for c in choices if c != 0]
    assert len(seen) == len(set(seen)), f"card re-enacted in {choices}"
    assert spec.forced_card_id not in seen, \
        f"forced card {spec.forced_card_id} is not choosable ({choices})"

    overrides = dict(spec.overrides)
    # window 0: the forced free card, baked into every branch
    forced = CARD_BY_ID[spec.forced_card_id]
    assert forced.key == "fund_repair", "forced-window wiring assumes fund_repair"
    plan: dict = {"repair_onset": spec.forced_tick,
                  "repair_spend_rate": REPAIR_SPEND_RATE,
                  "repair_efficiency": REPAIR_EFFICIENCY}
    debits = []
    mechs = []
    for tick, cid in zip(spec.windows, choices):
        if cid == 0:
            continue
        card = CARD_BY_ID[cid]
        if card.key == "ai_wealth_levy":
            plan.update(levy_onset=tick, levy_rate=LEVY_RATE)
            debits.append((tick, card.price))
        elif card.key == "sortition":
            mechs.append(scheduled(
                make_sortition(SortitionConfig(share=SORTITION_SHARE,
                                               adj_key="delegation")),
                cadence=SORTITION_CADENCE, phase_offset=tick, onset=tick))
            debits.append((tick, card.price))
        elif card.key == "influence_cap":
            overrides.update(reach_cut=REACH_CUT, reach_cut_onset=tick)
            debits.append((tick, card.price))
        elif card.key == "attention_campaign":
            plan.update(campaign_onset=tick, campaign_shift=CAMPAIGN_SHIFT)
        else:
            raise ValueError(card.key)
    interventions = make_interventions(
        InterventionPlanConfig(**plan, enforcement_debits=tuple(debits)))
    return overrides, (interventions, *mechs)


def path_id(choices: Tuple[int, ...]) -> str:
    """``p-<c1>-<c2>-<c3>`` — the tree's file and node naming."""
    return "p-" + "-".join(str(c) for c in choices)
