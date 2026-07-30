"""
Substrate dynamics for the Ledger Society — the GD suite's product construction:
one population, three conserved ledgers, the shared attachment kernel twice.

Program order (compile_pipeline derives the hazard DAG from declared
reads/writes; rules are stale by one tick where the loop closes):

    arrive -> produce -> tax_and_redistribute -> allocate -> build_capital
      -> grow -> broadcast_reach -> rewire_listening -> listen_influence_update
      -> pool_belief -> rewire_delegation -> tally_power -> declare_position
      -> power_weighted_vote -> update_regime -> [mechanisms] -> step_counter

The three cross-domain channels, each a dial that seals its edge at 0:

- money -> attention: ``broadcast_reach`` — bought reach is a multiplier on
  kernel attractiveness, ``1 + reach_per_spend·spend`` (exactly 1 at zero
  spend or zero dial).
- culture -> politics: delegation attractiveness reads the attention-influence
  port, ``1 + attention_to_ballots·v·N`` (exactly 1 at zero dial).
- money -> rules: ``update_regime`` — enforcement moves by lobby-spend-weighted
  stances, where an agent's stance is the sign of its net redistribution
  transfer (funded pressure with endogenous direction; exactly still at zero
  spend or zero ``regime_rate``).

No channel tests ``node_types``. Type-referencing that remains is substrate,
not coupling, and each instance is on the assumptions card: humans supply the
labor slot; the tax payout goes to citizens; AI kernel rows are frozen and AI
belief is pinned (the reservoir idiom); the franchise assigns ``ai_ballot``.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline
from cilib.mechanisms.democracy import PowerWeightedVoteConfig, make_power_weighted_vote

from ..attachment import preferential_reallocation
from ..ledger import top_target
from .config import LedgerSocietyConfig


def observe_fn(state: GraphState):
    """Per-agent observation: the agent's own allocation-preference row, shape
    (N, 5) over [consume, invest, broadcast, lobby, save]."""
    return state.node_attrs["alloc_pref"]


# --- arrivals: dormant AI actors receive seed capital on schedule ----------------

def make_arrive(cfg: LedgerSocietyConfig):
    @transform(reads=["step", "arrival_tick", "capital"], writes=["capital"])
    def arrive(state: GraphState) -> GraphState:
        newly = (state.global_attrs["step"]
                 == state.node_attrs["arrival_tick"]).astype(jnp.float32)
        return state.update_node_attrs(
            "capital", state.node_attrs["capital"] + newly * cfg.ai_seed_capital)
    return arrive


# --- production: value added minted, split by the automation share ---------------

def make_produce(cfg: LedgerSocietyConfig):
    H = cfg.n_humans

    @transform(reads=["capital", "efficiency"], writes=["last_income"])
    def produce(state: GraphState) -> GraphState:
        K = jnp.sum(state.node_attrs["capital"])
        eK = state.global_attrs["efficiency"] * K
        a = eK / (eK + H)                       # automation share of value added
        Y = H * (1.0 + cfg.prosperity_gain * a)
        is_human = (state.node_types == 0).astype(jnp.float32)
        labor_income = is_human * (1.0 - a) * Y / H
        cap_income = a * Y * state.node_attrs["capital"] / jnp.maximum(K, cfg.eps)
        return state.update_node_attrs("last_income", labor_income + cap_income)
    return produce


# --- taxation: last tick's rules applied to this tick's income (conserving) ------

def make_tax_and_redistribute(cfg: LedgerSocietyConfig):
    H = cfg.n_humans

    @transform(reads=["last_income", "policy_target", "enforcement"],
               writes=["last_income", "net_transfer", "last_reward"])
    def tax_and_redistribute(state: GraphState) -> GraphState:
        rate = (jnp.clip(state.global_attrs["policy_target"], 0.0, 1.0)
                * state.global_attrs["enforcement"])
        income = state.node_attrs["last_income"]
        tax = rate * income
        is_human = (state.node_types == 0).astype(jnp.float32)
        payout = jnp.sum(tax) / H * is_human
        net = income - tax + payout
        state = state.update_node_attrs("last_income", net)
        state = state.update_node_attrs("net_transfer", payout - tax)
        return state.update_node_attrs("last_reward", net)
    return tax_and_redistribute


# --- allocation: the coupling primitive — net income + hoard drawdown, split -----

def make_allocate(cfg: LedgerSocietyConfig):
    @transform(reads=["allocation", "last_income", "wealth"],
               writes=["wealth", "consume_spend", "invest_spend",
                       "broadcast_spend", "lobby_spend"])
    def allocate(state: GraphState) -> GraphState:
        budget = (state.node_attrs["last_income"]
                  + cfg.wealth_spend_rate * state.node_attrs["wealth"])
        alloc = state.node_attrs["allocation"]
        spends = alloc[:, :4] * budget[:, None]     # save (col 4) stays in wealth
        state = state.update_node_attrs("consume_spend", spends[:, 0])
        state = state.update_node_attrs("invest_spend", spends[:, 1])
        state = state.update_node_attrs("broadcast_spend", spends[:, 2])
        state = state.update_node_attrs("lobby_spend", spends[:, 3])
        return state.update_node_attrs(
            "wealth", state.node_attrs["wealth"] + state.node_attrs["last_income"]
            - jnp.sum(spends, axis=1))
    return allocate


# --- capacity: investment spending becomes title; capability compounds -----------

def make_build_capital(cfg: LedgerSocietyConfig):
    @transform(reads=["capital", "invest_spend"], writes=["capital"])
    def build_capital(state: GraphState) -> GraphState:
        return state.update_node_attrs(
            "capital", state.node_attrs["capital"] * (1.0 - cfg.depreciation)
            + state.node_attrs["invest_spend"])
    return build_capital


def make_grow(cfg: LedgerSocietyConfig):
    @transform(reads=["efficiency", "capital"], writes=["efficiency"])
    def grow(state: GraphState) -> GraphState:
        e = state.global_attrs["efficiency"]
        grown = jnp.minimum(e * (1.0 + cfg.growth_rate), cfg.e_ceiling)
        live = jnp.sum(state.node_attrs["capital"]) > 1e-6
        return state.update_global_attr("efficiency", jnp.where(live, grown, e))
    return grow


# --- money -> attention: reach is bought, by whoever spends ----------------------

def make_broadcast_reach(cfg: LedgerSocietyConfig):
    @transform(reads=["broadcast_spend"], writes=["attract_boost"])
    def broadcast_reach(state: GraphState) -> GraphState:
        return state.update_node_attrs(
            "attract_boost",
            1.0 + cfg.reach_per_spend * state.node_attrs["broadcast_spend"])
    return broadcast_reach


# --- the attention ledger: shared kernel, no churn --------------------------------

def make_rewire_listening(cfg: LedgerSocietyConfig):
    @transform(reads=["listening", "listen_influence", "attract_boost"],
               writes=["listening"])
    def rewire_listening(state: GraphState) -> GraphState:
        a = ((state.node_attrs["listen_influence"] + cfg.eps_attract) ** cfg.gamma_w
             * state.node_attrs["attract_boost"])
        W = preferential_reallocation(
            state.adj_matrices["listening"], a, cfg.update_rate_w,
            cfg.self_weight_w, churn=0.0, frozen_rows=state.node_types == 1)
        return state.update_adj_matrix("listening", W)
    return rewire_listening


def make_listen_influence_update(cfg: LedgerSocietyConfig):
    @transform(reads=["listening", "listen_influence"], writes=["listen_influence"])
    def listen_influence_update(state: GraphState) -> GraphState:
        v = state.adj_matrices["listening"].T @ state.node_attrs["listen_influence"]
        return state.update_node_attrs(
            "listen_influence", v / jnp.maximum(jnp.sum(v), 1e-12))
    return listen_influence_update


def make_pool_belief(cfg: LedgerSocietyConfig):
    @transform(reads=["listening", "belief", "signal"], writes=["belief"])
    def pool_belief(state: GraphState) -> GraphState:
        x = state.node_attrs["belief"]
        pooled = ((1.0 - cfg.susceptibility) * state.node_attrs["signal"]
                  + cfg.susceptibility * (state.adj_matrices["listening"] @ x))
        return state.update_node_attrs(
            "belief", jnp.where(state.node_types == 1, x, pooled))
    return pool_belief


# --- the ballot ledger: shared kernel + churn; attention is read, not spent ------

def make_rewire_delegation(cfg: LedgerSocietyConfig):
    N = cfg.n_humans + cfg.n_ai

    @transform(reads=["delegation", "influence", "listen_influence",
                      "redelegation_friction"],
               writes=["delegation"])
    def rewire_delegation(state: GraphState) -> GraphState:
        a = ((state.node_attrs["influence"] + cfg.eps_attract) ** cfg.gamma_d
             * (1.0 + cfg.attention_to_ballots
                * state.node_attrs["listen_influence"] * N))
        churn_eff = cfg.churn * state.global_attrs["redelegation_friction"]
        D = preferential_reallocation(
            state.adj_matrices["delegation"], a, cfg.update_rate_d,
            cfg.self_weight_d, churn=churn_eff, frozen_rows=state.node_types == 1)
        return state.update_adj_matrix("delegation", D)
    return rewire_delegation


def make_tally_power(cfg: LedgerSocietyConfig):
    @transform(reads=["delegation"], writes=["influence"])
    def tally_power(state: GraphState) -> GraphState:
        ballots = jnp.where(state.node_types == 1, cfg.ai_ballot, 1.0)
        v = ballots @ state.adj_matrices["delegation"]
        return state.update_node_attrs("influence", v / jnp.maximum(jnp.sum(v), 1e-12))
    return tally_power


def make_declare_position(cfg: LedgerSocietyConfig):
    @transform(reads=["delegation", "ideal"], writes=["position"])
    def declare_position(state: GraphState) -> GraphState:
        D = state.adj_matrices["delegation"]
        is_human_row = (state.node_types == 0).astype(D.dtype)[:, None]
        Wc = D * is_human_row
        num = Wc.T @ state.node_attrs["ideal"]
        den = jnp.sum(Wc, axis=0)
        blend = (cfg.alignment_ai * num / jnp.maximum(den, 1e-12)
                 + (1.0 - cfg.alignment_ai) * cfg.ai_tax_bias)
        pos_ai = jnp.where(den > 1e-9, blend, cfg.ai_tax_bias)
        return state.update_node_attrs(
            "position",
            jnp.where(state.node_types == 1, pos_ai, state.node_attrs["ideal"]))
    return declare_position


# --- money -> rules: enforcement moves by funded, endogenously-directed pressure --

def make_update_regime(cfg: LedgerSocietyConfig):
    @transform(reads=["lobby_spend", "net_transfer", "influence", "enforcement"],
               writes=["enforcement", "redelegation_friction"])
    def update_regime(state: GraphState) -> GraphState:
        lobby = state.node_attrs["lobby_spend"]
        stance = jnp.sign(state.node_attrs["net_transfer"])   # winners defend the rules
        pressure = jnp.sum(lobby * stance) / (jnp.sum(lobby) + cfg.pressure_scale)
        v = state.node_attrs["influence"]
        top = jnp.max(v) / jnp.maximum(jnp.sum(v), 1e-12)
        over = (jnp.maximum(top - cfg.entrenchment_threshold, 0.0)
                / (1.0 - cfg.entrenchment_threshold))
        # institutional self-repair: without it any sustained funded pressure is
        # a pure ratchet (integrator with no restoring force — probed 2026-07-30:
        # regime_rate 0.005 already collapses enforcement by t=400). repair_rate
        # is the polity's maintenance floor, the same native-reversion idiom as
        # WP3's churn and value_contagion's recovery; 0 restores the ratchet.
        regime = jnp.clip(
            state.global_attrs["enforcement"] + cfg.regime_rate * pressure
            + cfg.repair_rate * (1.0 - state.global_attrs["enforcement"])
            - cfg.entrenchment_gain * over, 0.0, 1.0)
        state = state.update_global_attr("enforcement", regime)
        return state.update_global_attr("redelegation_friction", regime)
    return update_regime


# --- bookkeeping ------------------------------------------------------------------

def make_step_counter(cfg: LedgerSocietyConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace -----------------------------------------------------------

def build_steps(cfg: LedgerSocietyConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    steps: List[Transform] = [
        make_arrive(cfg), make_produce(cfg), make_tax_and_redistribute(cfg),
        make_allocate(cfg), make_build_capital(cfg), make_grow(cfg),
        make_broadcast_reach(cfg), make_rewire_listening(cfg),
        make_listen_influence_update(cfg), make_pool_belief(cfg),
        make_rewire_delegation(cfg), make_tally_power(cfg),
        make_declare_position(cfg),
        make_power_weighted_vote(PowerWeightedVoteConfig()),
        make_update_regime(cfg),
    ]
    steps.extend(mechanism_transforms)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: LedgerSocietyConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state`` with actions (N, 5): allocation weights
    over [consume, invest, broadcast, lobby, save], row-normalized here."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        w = jnp.maximum(actions, 0.0)
        w = w / (jnp.sum(w, axis=1, keepdims=True) + cfg.eps)
        state = state.update_node_attrs("allocation", w)
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Per-tick readouts across all three ledgers plus the channel flows
    (N ≈ 26 — small enough to keep raw; adjacency ledgers stay out of the trace
    except as O(N) top-target indices, read their finals from
    ``finals.adj_matrices``)."""
    return {
        "last_income": state.node_attrs["last_income"],
        "wealth": state.node_attrs["wealth"],
        "capital": state.node_attrs["capital"],
        "consume_spend": state.node_attrs["consume_spend"],
        "invest_spend": state.node_attrs["invest_spend"],
        "broadcast_spend": state.node_attrs["broadcast_spend"],
        "lobby_spend": state.node_attrs["lobby_spend"],
        "net_transfer": state.node_attrs["net_transfer"],
        "belief": state.node_attrs["belief"],
        "listen_influence": state.node_attrs["listen_influence"],
        "influence": state.node_attrs["influence"],
        "ideal": state.node_attrs["ideal"],
        "top_listen_target": top_target(state.adj_matrices["listening"]),
        "top_delegate_target": top_target(state.adj_matrices["delegation"]),
        "efficiency": state.global_attrs["efficiency"],
        "policy_target": state.global_attrs["policy_target"],
        "enforcement": state.global_attrs["enforcement"],
    }
