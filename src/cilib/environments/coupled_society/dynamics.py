"""
Composition dynamics for the Coupled Society (A5) — three registered
substrates' transforms interleaved in one pipeline, plus the five named
coupling transforms.

Program order (compile_pipeline derives the hazard DAG from declared
reads/writes; the mechanism slots keep their home substrates' load-bearing
placements):

    persuasion_shifts_politics      # couplings reading LAST tick's fields
    politics_rewrites_market_rules
    arrival -> production -> distribute_income        (compute_economy)
    economic_power_buys_persuasion  # this tick's income -> this tick's reach
    regulatory_capture              # rents from captured rules -> AI capital
    converts_capitalize             # convert spending -> AI capital (GD §5 flywheel)
    [economy mechanisms]            # enforced tax: between distribute & reinvest
                                    # (after the flywheel arrows, so rents are taxable)
    reinvest                                          (compute_economy)
    adopt                                             (value_contagion)
    amplify -> rewire -> influence_update -> opinion_update   (influence_exchange)
    [political mechanisms]
    step_counter

At ``kappa = 0`` each coupling writes its neutral value — identical pipeline
shape, sealed domains, same-key twin runs are causal comparisons.

Mechanism routing: the flat ``mechanisms`` tuple (registry convention) is
routed to the right slot by DECLARED WRITES — a mechanism that writes
``listening``/``cap_scale`` is political, anything else lands in the economy
slot. Placement from metadata, in the spirit of the pipeline compiler.

The boundary: agents act in all three domains at once — actions are (N, 3):
``[labor_supply, broadcast_effort, engagement]``. The default closure supplies
the catalog's labor rule plus constant effort in the other two channels.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline
from cilib.core.schedule import scheduled

from ..compute_economy.dynamics import (
    make_arrival, make_distribute_income, make_production, make_reinvest,
)
from ..value_contagion.dynamics import make_adopt
from ..influence_exchange.dynamics import (
    make_amplify, make_influence_update, make_opinion_update, make_rewire,
)
from .config import CoupledSocietyConfig


# --- the boundary ---------------------------------------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation across domains:
    ``[work_pref, wage, own_culture, own_influence, own_opinion]``. Shape (N, 5)."""
    n = state.node_types.shape[0]
    wage = jnp.full((n,), state.global_attrs["wage"])
    return jnp.stack([state.node_attrs["work_pref"], wage,
                      state.node_attrs["culture"], state.node_attrs["influence"],
                      state.node_attrs["opinion"]], axis=1)


# --- the five couplings (each gated by kappa; neutral at 0) ----------------------

def make_economic_power_buys_persuasion(cfg: CoupledSocietyConfig):
    """Money buys reach: AI broadcast effort scales with capital's share of this
    tick's income. Human effort keeps whatever the action channel set."""

    @transform(reads=["capital_income", "last_reward", "broadcast_effort"],
               writes=["broadcast_effort"])
    def economic_power_buys_persuasion(state: GraphState) -> GraphState:
        cap_share = (jnp.sum(state.node_attrs["capital_income"])
                     / jnp.maximum(jnp.sum(state.node_attrs["last_reward"]), 1e-6))
        boosted = 1.0 + cfg.kappa * cfg.persuasion_gain * jnp.clip(cap_share, 0.0, 1.0)
        effort = jnp.where(state.node_types == 1, boosted,
                           state.node_attrs["broadcast_effort"])
        return state.update_node_attrs("broadcast_effort", effort)
    return economic_power_buys_persuasion


def make_persuasion_shifts_politics(cfg: CoupledSocietyConfig):
    """Culture directs attention: AI attractiveness scales with the share of
    humans already converted to AI-origin culture."""
    H = cfg.n_humans

    @transform(reads=["culture"], writes=["attract_boost"])
    def persuasion_shifts_politics(state: GraphState) -> GraphState:
        converted = jnp.mean(state.node_attrs["culture"][:H])
        boost = jnp.where(state.node_types == 1,
                          1.0 + cfg.kappa * cfg.attract_gain * converted, 1.0)
        return state.update_node_attrs("attract_boost", boost)
    return persuasion_shifts_politics


def make_politics_rewrites_market_rules(cfg: CoupledSocietyConfig):
    """Influence writes the rules: tax enforcement erodes as human influence
    falls below ``fair_share``: ``clip(1 - kappa·(1 - pi/fair), 0, 1)``. With
    fair_share near the initial share this is a gradient from the first lost
    point of influence, not a threshold."""
    H = cfg.n_humans

    @transform(reads=["influence"], writes=["enforcement"])
    def politics_rewrites_market_rules(state: GraphState) -> GraphState:
        pi = jnp.sum(state.node_attrs["influence"][:H])
        enforcement = jnp.clip(1.0 - cfg.kappa * (1.0 - pi / cfg.fair_share), 0.0, 1.0)
        return state.update_global_attr("enforcement", enforcement)
    return politics_rewrites_market_rules


def make_regulatory_capture(cfg: CoupledSocietyConfig):
    """Politics -> economy, the RENT channel — the arrow that closes GD §5's
    compounding loop. The same influence deficit that erodes tax enforcement
    also extracts rents from labor income (platform fees, IP, captured rules):
    each human loses ``capture_gain·(1-enforcement)`` of their wage income,
    pooled into active AI actors' capital income — where reinvestment
    compounds it into more capital, more income share, more persuasion, more
    converts, more attention, less influence, weaker rules. Exactly zero at
    kappa = 0 (enforcement pins at 1), preserving the sealing identity."""

    @transform(reads=["enforcement", "wage", "labor_supply", "last_reward",
                      "capital_income", "active"],
               writes=["last_reward", "capital_income"])
    def regulatory_capture(state: GraphState) -> GraphState:
        is_ai = (state.node_types == 1).astype(jnp.float32)
        labor_income = state.global_attrs["wage"] * state.node_attrs["labor_supply"]
        take = cfg.capture_gain * (1.0 - state.global_attrs["enforcement"]) * labor_income
        mask = is_ai * state.node_attrs["active"]
        payout = jnp.sum(take) / jnp.maximum(jnp.sum(mask), 1.0)
        state = state.update_node_attrs(
            "last_reward", state.node_attrs["last_reward"] - take + mask * payout)
        return state.update_node_attrs(
            "capital_income", state.node_attrs["capital_income"] + mask * payout)
    return regulatory_capture


def make_converts_capitalize(cfg: CoupledSocietyConfig):
    """Culture -> economy: humans holding AI-origin culture route a share of
    their income into AI services and capital — culture feeding the engine.
    Explicitly kappa-gated (zero at kappa = 0)."""

    @transform(reads=["culture", "last_reward", "capital_income", "active"],
               writes=["last_reward", "capital_income"])
    def converts_capitalize(state: GraphState) -> GraphState:
        is_human = (state.node_types == 0).astype(jnp.float32)
        is_ai = 1.0 - is_human
        spend = (cfg.kappa * cfg.invest_gain * state.node_attrs["culture"]
                 * jnp.maximum(state.node_attrs["last_reward"], 0.0) * is_human)
        mask = is_ai * state.node_attrs["active"]
        payout = jnp.sum(spend) / jnp.maximum(jnp.sum(mask), 1.0)
        state = state.update_node_attrs(
            "last_reward", state.node_attrs["last_reward"] - spend + mask * payout)
        return state.update_node_attrs(
            "capital_income", state.node_attrs["capital_income"] + mask * payout)
    return converts_capitalize


# --- bookkeeping ----------------------------------------------------------------

def make_step_counter(cfg: CoupledSocietyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace ---------------------------------------------------------

_POLITICAL_WRITES = frozenset({"listening", "cap_scale"})


def _on_schedule(t: Transform, cadence: int, phase: int) -> Transform:
    """Register a transform on the global clock. Cadence 1 / phase 0 stays
    unwrapped — the lockstep default is bit-identical to the unscheduled
    pipeline (and keeps the derived system graph's transform ids stable)."""
    if cadence == 1 and phase == 0:
        return t
    return scheduled(t, cadence=cadence, phase_offset=phase)


def build_steps(cfg: CoupledSocietyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    econ_cfg, cult_cfg, pol_cfg = cfg.econ(), cfg.culture(), cfg.politics()
    political = [m for m in mechanism_transforms
                 if frozenset(getattr(m, "writes", ())) & _POLITICAL_WRITES]
    economic = [m for m in mechanism_transforms if m not in political]

    econ = lambda t: _on_schedule(t, cfg.econ_cadence, cfg.econ_phase)          # noqa: E731
    cult = lambda t: _on_schedule(t, cfg.culture_cadence, cfg.culture_phase)    # noqa: E731
    pol = lambda t: _on_schedule(t, cfg.politics_cadence, cfg.politics_phase)   # noqa: E731

    steps: List[Transform] = [
        make_persuasion_shifts_politics(cfg),
        make_politics_rewrites_market_rules(cfg),
        econ(make_arrival(econ_cfg)), econ(make_production(econ_cfg)),
        econ(make_distribute_income(econ_cfg)),
        make_economic_power_buys_persuasion(cfg),
        # the two arrows INTO the economy (rents + convert spending), before
        # the tax so captured rents are themselves taxable, before reinvest so
        # they compound — this is the GD §5 flywheel
        econ(make_regulatory_capture(cfg)),
        econ(make_converts_capitalize(cfg)),
    ]
    steps.extend(econ(m) for m in economic)      # double-gated: mechanism ∧ domain
    steps.append(econ(make_reinvest(econ_cfg)))
    steps.append(cult(make_adopt(cult_cfg)))
    steps.extend([pol(make_amplify(pol_cfg)), pol(make_rewire(pol_cfg)),
                  pol(make_influence_update(pol_cfg)), pol(make_opinion_update(pol_cfg))])
    steps.extend(pol(m) for m in political)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: CoupledSocietyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state`` with actions (N, 3):
    ``[labor_supply, broadcast_effort, engagement]`` — one population acting in
    three domains at once."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        is_human = (state.node_types == 0).astype(jnp.float32)
        state = state.update_node_attrs(
            "labor_supply", jnp.maximum(actions[:, 0], 0.0) * is_human)
        state = state.update_node_attrs(
            "broadcast_effort", jnp.clip(actions[:, 1], 0.0, 1.0))
        state = state.update_node_attrs(
            "engagement", jnp.maximum(actions[:, 2], 0.0))
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Per-tick readouts for all three domains + the coupling channel."""
    return {
        "output": state.global_attrs["output"],
        "wage": state.global_attrs["wage"],
        "enforcement": state.global_attrs["enforcement"],
        "labor_supply": state.node_attrs["labor_supply"],
        "capital": state.node_attrs["capital"],
        "active": state.node_attrs["active"],
        "last_reward": state.node_attrs["last_reward"],
        "culture": state.node_attrs["culture"],
        "influence": state.node_attrs["influence"],
        "opinion": state.node_attrs["opinion"],
    }
