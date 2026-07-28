"""Mechanisms catalog — composed institutions as typed transforms.

Type function:  ``TransformFactory = Config -> Transform``. A mechanism is a Transform
that declares its ``.reads`` / ``.writes`` (via ``@transform``) so ``compile_pipeline``
can order it; entries of the same *family* keep disjoint write sets so ``parallel``
composition is always valid.

Families (Plan 2): ``market`` · ``network`` · ``democracy``. Seeded today: ``market``
and the ``democracy`` pair (``quota_vote``, ``graduated_sanction``) — the variants
(double-auction, gossip, liquid democracy …) are the first open-source follow-ups.
See README.md for the contract.
"""
from .market import create_market_transform
from .democracy import (
    QuotaVoteConfig, SanctionConfig, make_quota_vote, make_graduated_sanction,
)
from .fiscal import (
    AIRevenueTaxConfig, OwnershipCapConfig, make_ai_revenue_tax, make_ownership_cap,
)
from .political import (
    SortitionConfig, InfluenceCapConfig, make_sortition, make_influence_cap,
)

# name -> factory ((cfg) -> Transform).
REGISTRY = {
    "market": create_market_transform,
    "quota_vote": make_quota_vote,
    "graduated_sanction": make_graduated_sanction,
    "ai_revenue_tax": make_ai_revenue_tax,
    "ownership_cap": make_ownership_cap,
    "sortition": make_sortition,
    "influence_cap": make_influence_cap,
}

__all__ = [
    "create_market_transform",
    "QuotaVoteConfig", "SanctionConfig", "make_quota_vote", "make_graduated_sanction",
    "AIRevenueTaxConfig", "OwnershipCapConfig", "make_ai_revenue_tax", "make_ownership_cap",
    "SortitionConfig", "InfluenceCapConfig", "make_sortition", "make_influence_cap",
    "REGISTRY",
]
