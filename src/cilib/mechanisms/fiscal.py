"""
Fiscal family — taxation and ownership institutions (IAD payoff/boundary rules).

Two composable entries, contracts kept generic so any economy-shaped environment
exposing the fields can use them:

- ``ai_revenue_tax`` (payoff rule): taxes ``capital_income`` at a flat rate and
  redistributes the proceeds equally to active human (type-0) nodes. Tax + redistribute
  are one atomic mechanism (like ``graduated_sanction`` bundles penalize + confiscate) —
  splitting them would force a ``last_reward`` write overlap and break the same-family
  disjoint-writes invariant. **Placement is load-bearing**: composed between income
  distribution and reinvestment (the environment's mechanism slot), the tax reduces what
  gets reinvested and thereby slows compute compounding — spliced after reinvestment it
  would change take-home pay without touching the dynamics at all.
- ``ownership_cap`` (boundary rule): caps any single actor's share of aggregate active
  AI capital *before* production. Single-pass (not fixed-point iterated) — a documented
  v0 simplification; excess capital simply evaporates rather than being redistributed.

Disjoint family writes: tax → {capital_income, last_reward}; cap → {capital}. Both
compose in one mechanism list, order-independent. Timing belongs to the schedule
(``core.schedule.scheduled`` — e.g. a tax switching on at t=100 is the regime-shift dial).
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform


@dataclasses.dataclass(frozen=True)
class AIRevenueTaxConfig:
    tax_rate: float = 0.5    # flat rate on capital income


def make_ai_revenue_tax(cfg: AIRevenueTaxConfig):
    """Tax capital income, redistribute equally to active human (type-0) nodes."""

    @transform(reads=["capital_income", "active", "last_reward"],
               writes=["capital_income", "last_reward"])
    def ai_revenue_tax(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        recipients = is_household * state.node_attrs["active"]
        tax = cfg.tax_rate * state.node_attrs["capital_income"]
        payout = jnp.sum(tax) / (jnp.sum(recipients) + 1e-8)
        state = state.update_node_attrs(
            "capital_income", state.node_attrs["capital_income"] - tax)
        return state.update_node_attrs(
            "last_reward", state.node_attrs["last_reward"] - tax + recipients * payout)
    return ai_revenue_tax


@dataclasses.dataclass(frozen=True)
class EnforcedAITaxConfig:
    tax_rate: float = 0.5    # statutory rate; effective rate = tax_rate x enforcement


def make_enforced_ai_tax(cfg: EnforcedAITaxConfig):
    """``ai_revenue_tax`` whose EFFECTIVE rate is scaled by the environment's
    ``enforcement`` global (in [0, 1]) — the rule-on-paper vs rule-in-practice
    distinction. In an environment holding ``enforcement`` at 1 this is exactly
    the flat tax; in coupled_society, ``politics_rewrites_market_rules`` erodes
    it as human influence falls (Gradual Disempowerment §5's shifted burdens).
    ``validate_reads`` rejects composition into any environment that has no
    ``enforcement`` field — by design, at build time."""

    @transform(reads=["capital_income", "active", "last_reward", "enforcement"],
               writes=["capital_income", "last_reward"])
    def enforced_ai_tax(state: GraphState) -> GraphState:
        is_household = (state.node_types == 0).astype(jnp.float32)
        recipients = is_household * state.node_attrs["active"]
        rate = cfg.tax_rate * state.global_attrs["enforcement"]
        tax = rate * state.node_attrs["capital_income"]
        payout = jnp.sum(tax) / (jnp.sum(recipients) + 1e-8)
        state = state.update_node_attrs(
            "capital_income", state.node_attrs["capital_income"] - tax)
        return state.update_node_attrs(
            "last_reward", state.node_attrs["last_reward"] - tax + recipients * payout)
    return enforced_ai_tax


@dataclasses.dataclass(frozen=True)
class OwnershipCapConfig:
    cap_share: float = 0.35  # max fraction of aggregate active AI capital per actor


def make_ownership_cap(cfg: OwnershipCapConfig):
    """Cap any single AI (type-1) actor's share of aggregate active AI capital."""

    @transform(reads=["capital", "active"], writes=["capital"])
    def ownership_cap(state: GraphState) -> GraphState:
        is_ai = (state.node_types == 1).astype(jnp.float32)
        mask = is_ai * state.node_attrs["active"]
        total = jnp.sum(state.node_attrs["capital"] * mask) + 1e-8
        capped = jnp.minimum(state.node_attrs["capital"], cfg.cap_share * total)
        return state.update_node_attrs(
            "capital", jnp.where(mask > 0, capped, state.node_attrs["capital"]))
    return ownership_cap
