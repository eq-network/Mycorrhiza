"""The GD game's three lever families — economy (allocate), culture (shape),
politics (spend). One composed transform per family, one ``(T, P)`` plan array
per family carried in ``global_attrs`` as a dynamic pytree child, so plan
*values* never force a recompile. See docs/gd-game-three-families.md and
README.md.

**The write map** (checked by ``tests/test_families.py``; the ordering the
pipeline compiler derives from it is economy → politics, with culture free to
batch alongside either):

    economy   alloc_pref, wealth
    culture   gamma_w_now, update_rate_w_now, reach_cut_now
    politics  wealth, intervention_spend, enforcement, delegation,
              repair_rate_now, entrenchment_gain_now

Culture is disjoint from both siblings — it writes only the substrate globals
the attention path reads next tick. Economy and politics share exactly one
field, ``wealth``, and that overlap is **deliberate, not a naming accident**:
the levy is a conserving AI→human transfer of the stock and the office drip
removes part of the stock into the declared ``intervention_spend`` sink. They
are two different acts on the same money, and the spec's third failure mode is
precisely that trade — funding the office competes with investing the hoard.
Neither family can give the field up without inventing a second money ledger.
``compile_pipeline`` turns the shared write into a plain WAW edge oriented by
program order, so the composition is valid and deterministic: pass the
mechanisms in the order ``(economy, culture, politics)`` and the drip is taken
from post-levy wealth. Because of the overlap the three are **not** a
``parallel``-composable family in the catalog sense; they are one bundle
composed in sequence.

Enforcement has exactly one writer, politics. The other two families' political
acts — economy's levy, culture's reach cap — are billed to that stock as data
through the ``external_intensity`` plan column, priced by
``politics.external_intensity_of`` off the siblings' own plan rows.

All three are alternative closures to ``make_policy_levers`` and
``make_interventions``, never composable with them: the card game, the
four-lever policy game and the three-family game each own the same channels.
"""
from .culture import (
    CULTURE_LEVERS, CULTURE_PLAN, CultureLeverConfig, culture_upkeep,
    make_culture_levers, neutral_culture_plan, attach_culture_plan,
)
from .economy import (
    ECONOMY_LEVERS, ECONOMY_PLAN, EconomyLeverConfig, make_economy_levers,
    neutral_economy_plan, attach_economy_plan,
)
from .politics import (
    POLITICS_LEVERS, POLITICS_PLAN, PoliticsLeverConfig, make_politics_levers,
    neutral_politics_plan, attach_politics_plan, external_intensity_of,
    enforcement_rest,
)

# name -> factory ((cfg) -> Transform); mirrored into cilib.mechanisms.REGISTRY
FAMILY_FACTORIES = {
    "economy_levers": make_economy_levers,
    "culture_levers": make_culture_levers,
    "politics_levers": make_politics_levers,
}

__all__ = [
    "CULTURE_LEVERS", "CULTURE_PLAN", "CultureLeverConfig", "culture_upkeep",
    "make_culture_levers", "neutral_culture_plan", "attach_culture_plan",
    "ECONOMY_LEVERS", "ECONOMY_PLAN", "EconomyLeverConfig",
    "make_economy_levers", "neutral_economy_plan", "attach_economy_plan",
    "POLITICS_LEVERS", "POLITICS_PLAN", "PoliticsLeverConfig",
    "make_politics_levers", "neutral_politics_plan", "attach_politics_plan",
    "external_intensity_of", "enforcement_rest",
    "FAMILY_FACTORIES",
]
