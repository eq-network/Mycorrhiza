"""Mechanisms catalog — composed institutions as typed transforms.

Type function:  ``TransformFactory = Config -> Transform``. A mechanism is a Transform
that declares its ``.reads`` / ``.writes`` (via ``@transform``) so ``compile_pipeline``
can order it; entries of the same *family* keep disjoint write sets so ``parallel``
composition is always valid.

Families (Plan 2): ``market`` · ``network`` · ``democracy``. Seeded today: ``market``
and the ``democracy`` pair (``quota_vote``, ``graduated_sanction``) — the variants
(double-auction, gossip, liquid democracy …) are the first open-source follow-ups.
See README.md for the contract.

``families/`` holds the GD game's three lever bundles — ``economy_levers``,
``culture_levers``, ``politics_levers`` — each one composed transform reading one
plan array. They are a *sequenced bundle*, not a parallel family: economy and
politics share the ``wealth`` write on purpose. See families/README.md.
"""
from .market import create_market_transform
from .democracy import (
    QuotaVoteConfig, SanctionConfig, PowerWeightedVoteConfig,
    make_quota_vote, make_graduated_sanction, make_power_weighted_vote,
)
from .fiscal import (
    AIRevenueTaxConfig, EnforcedAITaxConfig, OwnershipCapConfig,
    make_ai_revenue_tax, make_enforced_ai_tax, make_ownership_cap,
)
from .political import (
    SortitionConfig, InfluenceCapConfig, make_sortition, make_influence_cap,
)
from .interventions import (
    NEVER, PLAN_LEVERS, InterventionPlanConfig, PolicyLeverConfig,
    make_interventions, make_policy_levers,
)
from .families import (
    ECONOMY_LEVERS, ECONOMY_PLAN, EconomyLeverConfig, make_economy_levers,
    neutral_economy_plan, attach_economy_plan,
    CULTURE_LEVERS, CULTURE_PLAN, CultureLeverConfig, make_culture_levers,
    neutral_culture_plan, attach_culture_plan, culture_upkeep,
    POLITICS_LEVERS, POLITICS_PLAN, PoliticsLeverConfig, make_politics_levers,
    neutral_politics_plan, attach_politics_plan, external_intensity_of,
    enforcement_rest,
)

# name -> factory ((cfg) -> Transform).
REGISTRY = {
    "market": create_market_transform,
    "quota_vote": make_quota_vote,
    "graduated_sanction": make_graduated_sanction,
    "power_weighted_vote": make_power_weighted_vote,
    "ai_revenue_tax": make_ai_revenue_tax,
    "enforced_ai_tax": make_enforced_ai_tax,
    "ownership_cap": make_ownership_cap,
    "sortition": make_sortition,
    "influence_cap": make_influence_cap,
    "interventions": make_interventions,
    "policy_levers": make_policy_levers,
    "economy_levers": make_economy_levers,
    "culture_levers": make_culture_levers,
    "politics_levers": make_politics_levers,
}

__all__ = [
    "create_market_transform",
    "QuotaVoteConfig", "SanctionConfig", "make_quota_vote", "make_graduated_sanction",
    "PowerWeightedVoteConfig", "make_power_weighted_vote",
    "AIRevenueTaxConfig", "OwnershipCapConfig", "make_ai_revenue_tax", "make_ownership_cap",
    "EnforcedAITaxConfig", "make_enforced_ai_tax",
    "SortitionConfig", "InfluenceCapConfig", "make_sortition", "make_influence_cap",
    "NEVER", "PLAN_LEVERS", "InterventionPlanConfig", "PolicyLeverConfig",
    "make_interventions", "make_policy_levers",
    "ECONOMY_LEVERS", "ECONOMY_PLAN", "EconomyLeverConfig", "make_economy_levers",
    "neutral_economy_plan", "attach_economy_plan",
    "CULTURE_LEVERS", "CULTURE_PLAN", "CultureLeverConfig", "make_culture_levers",
    "neutral_culture_plan", "attach_culture_plan", "culture_upkeep",
    "POLITICS_LEVERS", "POLITICS_PLAN", "PoliticsLeverConfig",
    "make_politics_levers", "neutral_politics_plan", "attach_politics_plan",
    "external_intensity_of", "enforcement_rest",
    "REGISTRY",
]
