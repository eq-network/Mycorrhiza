"""
Substrate dynamics for Delegative Polity — one information-flow loop from
citizen preferences to enacted tax policy and back through the delegation
graph. OPEN at the agent boundary.

Round (post-action pipeline, composed via ``compile_pipeline``):

    amplify -> tally_power -> declare_position -> power_weighted_vote
      -> tax_and_redistribute -> update_regime -> rewire_delegation
      -> [mechanisms] -> counter

- **amplify**: pure function of the clock — AI delegates' attractiveness
  multiplier is ``ai_advantage`` from ``ai_advantage_onset``, 1 before (the
  compute_economy arrival idiom: the threat is a schedule, not a decision).
- **tally_power**: one-hop ballot weight ``v = colsum(D) / N`` — everyone
  starts the tick with one vote; your power is the share of ballots in your
  hand after delegation (a delegate casts what it holds, it does not forward).
- **declare_position**: citizens declare their own fixed ``ideal`` (a human
  super-voter votes THEIR ideal, not their delegators' — faithful liquid
  democracy, itself a quiet disempowerment channel); AI delegates declare
  ``alignment_ai * (delegators' weighted mean ideal) + (1-alignment_ai) * ai_bias``
  (agents/delegate.py's fidelity blend, inlined).
- **power_weighted_vote** (mechanisms/democracy.py): the power-weighted median
  of positions becomes ``policy_target`` — the institution itself, hardwired.
- **tax_and_redistribute**: effective rate = ``policy_target x enforcement``;
  flat tax on citizen endowments, equal per-citizen payout. Money-conserving
  by construction; ``last_reward`` is the per-tick net income (GameSpec hook).
- **update_regime**: the lock-in. One ``regime`` scalar falls smoothly as the
  top node's power share exceeds ``entrenchment_threshold`` and gates BOTH
  ``enforcement`` and ``redelegation_friction``. ``entrenchment_gain=0`` pins
  it at exactly 1.0 forever — the honest region. This round's power writes
  next round's rules (Acemoglu-Robinson's de-facto -> de-jure loop); there is
  no hard-coded ratchet, so any hysteresis must EMERGE from the feedback.
- **rewire_delegation** (preferential attachment vs churn): each citizen row
  drifts ``update_rate`` toward the attractiveness distribution and
  ``churn x redelegation_friction`` toward a uniform re-draw (the freedom to
  re-delegate). Rows scale with the citizen's own ``engagement``; a row with
  no off-diagonal mass (a pure self-voter) is a fixed point — nothing to
  redistribute. AI rows are frozen; AI power is being delegated TO.

The boundary: agents observe ``[influence, wealth]`` and choose ``engagement``
(how actively they compete for and reconsider delegation); the catalog's
constant-effort ``BroadcastPolicy`` closes it (same boundary as
influence_exchange, deliberately).
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline
from cilib.mechanisms.democracy import PowerWeightedVoteConfig, make_power_weighted_vote

from ..ledger import top_target
from .config import DelegativePolityConfig


# --- the boundary: what an agent observes --------------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[own_influence, own_wealth]``. Shape (N, 2)."""
    return jnp.stack([state.node_attrs["influence"],
                      state.node_attrs["wealth"]], axis=1)


# --- scheduled attractiveness advantage of AI delegates --------------------------

def make_amplify(cfg: DelegativePolityConfig):
    @transform(reads=["step"], writes=["amplification"])
    def amplify(state: GraphState) -> GraphState:
        step = state.global_attrs["step"]
        on = (step >= cfg.ai_advantage_onset) & (step < cfg.ai_advantage_offset)
        amp = jnp.where((state.node_types == 1) & on, cfg.ai_advantage, 1.0)
        return state.update_node_attrs("amplification", amp)
    return amplify


# --- power: ballots received in one delegation hop -------------------------------

def make_tally_power(cfg: DelegativePolityConfig):
    """One-hop ballot weight: every ballot-holder starts each tick with their
    base vote (1 for citizens, ``ai_ballot`` for AI delegates); your power is
    the share of votes in your hand after delegation — your kept diagonal plus
    what others handed you (ballot-weighted column sums of D, normalized).
    ``ai_ballot`` is a declared power floor: at the default 1.0, AI delegates
    hold votes of their own and their frozen uniform rows hand most of that
    mass back to citizens every tick; at 0 they are pure conduits.
    Deliberately NOT the eigenvector: a delegate CASTS the ballots it holds
    rather than forwarding them forever, and the recycling fixed point would
    cap the AI bloc below a majority by construction (transitive/spectral
    voice is the deferred extension, stated on the card)."""

    @transform(reads=["delegation"], writes=["influence"])
    def tally_power(state: GraphState) -> GraphState:
        D = state.adj_matrices["delegation"]
        ballots = jnp.where(state.node_types == 1, cfg.ai_ballot, 1.0)
        v = ballots @ D
        return state.update_node_attrs("influence", v / jnp.maximum(jnp.sum(v), 1e-12))
    return tally_power


# --- positions: own ideal for citizens, the fidelity blend for AI ----------------

def make_declare_position(cfg: DelegativePolityConfig):
    @transform(reads=["delegation", "ideal"], writes=["position"])
    def declare_position(state: GraphState) -> GraphState:
        D = state.adj_matrices["delegation"]
        is_cit_row = (state.node_types == 0).astype(D.dtype)[:, None]
        Wc = D * is_cit_row                       # who delegates: citizen rows only
        num = Wc.T @ state.node_attrs["ideal"]
        den = jnp.sum(Wc, axis=0)
        blend = (cfg.alignment_ai * num / jnp.maximum(den, 1e-12)
                 + (1.0 - cfg.alignment_ai) * cfg.ai_bias)
        pos_ai = jnp.where(den > 1e-9, blend, cfg.ai_bias)   # no delegators: own pull
        return state.update_node_attrs(
            "position",
            jnp.where(state.node_types == 1, pos_ai, state.node_attrs["ideal"]))
    return declare_position


# --- taxation: flat rate x enforcement, equal redistribution ---------------------

def make_tax_and_redistribute(cfg: DelegativePolityConfig):
    @transform(reads=["endowment", "wealth", "policy_target", "enforcement"],
               writes=["wealth", "last_reward"])
    def tax_and_redistribute(state: GraphState) -> GraphState:
        is_cit = (state.node_types == 0).astype(jnp.float32)
        r_eff = (jnp.clip(state.global_attrs["policy_target"], 0.0, 1.0)
                 * state.global_attrs["enforcement"])
        endow = state.node_attrs["endowment"]
        tax = r_eff * endow * is_cit
        payout = jnp.sum(tax) / cfg.n_citizens
        net = (endow - tax + payout) * is_cit
        state = state.update_node_attrs("last_reward", net)
        return state.update_node_attrs("wealth", state.node_attrs["wealth"] + net)
    return tax_and_redistribute


# --- lock-in: concentrated power erodes the rules themselves ---------------------

def make_update_regime(cfg: DelegativePolityConfig):
    @transform(reads=["influence"], writes=["enforcement", "redelegation_friction"])
    def update_regime(state: GraphState) -> GraphState:
        v = state.node_attrs["influence"]
        top = jnp.max(v) / jnp.maximum(jnp.sum(v), 1e-12)
        over = (jnp.maximum(top - cfg.entrenchment_threshold, 0.0)
                / (1.0 - cfg.entrenchment_threshold))
        regime = jnp.clip(1.0 - cfg.entrenchment_gain * over, 0.0, 1.0)
        state = state.update_global_attr("enforcement", regime)
        return state.update_global_attr("redelegation_friction", regime)
    return update_regime


# --- preferential attachment vs churn on the delegation matrix -------------------

def make_rewire_delegation(cfg: DelegativePolityConfig):
    """CONSCIOUS FORK of ``influence_exchange.make_rewire`` — the suite's shared
    attachment kernel (docs/gd-suite-v0.1.md §2.4), plus exactly three declared
    differences: churn toward a uniform re-draw, the (erodible) franchise floor
    on the diagonal, and engagement-gated rows. Any further divergence between
    the two copies is a suite-level bug, not a local edit."""

    @transform(reads=["delegation", "influence", "amplification", "cap_scale",
                      "attract_boost", "engagement", "redelegation_friction"],
               writes=["delegation"])
    def rewire_delegation(state: GraphState) -> GraphState:
        W = state.adj_matrices["delegation"]
        N = W.shape[0]
        eye = jnp.eye(N)

        a = ((state.node_attrs["influence"] + cfg.eps_attract) ** cfg.gamma
             * state.node_attrs["amplification"]
             * state.node_attrs["cap_scale"]
             * state.node_attrs["attract_boost"]
             * state.node_attrs["engagement"])
        target = a[None, :] * (1.0 - eye)                    # never toward self
        t_mass = jnp.sum(target, axis=1, keepdims=True)
        target = target / jnp.maximum(t_mass, 1e-12)

        uniform = (1.0 - eye) / (N - 1)                      # the re-draw target
        friction = state.global_attrs["redelegation_friction"]
        r_eff = cfg.churn * friction
        # the franchise floor: by default the kept-vote share is untouchable;
        # with franchise_erosion armed, a dead regime makes the vote itself
        # delegatable (lock-in's third channel — see config docstring)
        s_w = cfg.self_weight * (1.0 - cfg.franchise_erosion * (1.0 - friction))

        offdiag = W * (1.0 - eye)
        off_mass = jnp.sum(offdiag, axis=1, keepdims=True)
        off_norm = offdiag / jnp.maximum(off_mass, 1e-12)
        drift = ((1.0 - cfg.update_rate - r_eff) * off_norm
                 + cfg.update_rate * target + r_eff * uniform)
        # a row with nowhere to drift (all-silent world) keeps its delegation
        mixed = jnp.where(t_mass > 1e-9, drift, off_norm)
        W_new = s_w * eye + (1.0 - s_w) * mixed

        # a pure self-voter has no off-diagonal mass to redistribute: fixed point
        W_new = jnp.where(off_mass > 1e-9, W_new, W)
        # re-delegation is an act: a disengaged citizen's row does not move
        act = jnp.clip(state.node_attrs["engagement"], 0.0, 1.0)[:, None]
        W_new = act * W_new + (1.0 - act) * W

        is_ai_row = (state.node_types == 1)[:, None]
        return state.update_adj_matrix("delegation", jnp.where(is_ai_row, W, W_new))
    return rewire_delegation


# --- bookkeeping ----------------------------------------------------------------

def make_step_counter(cfg: DelegativePolityConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace ---------------------------------------------------------

def build_steps(cfg: DelegativePolityConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline in program order: substrate (including the voting
    institution), defense mechanisms, counter."""
    steps: List[Transform] = [
        make_amplify(cfg), make_tally_power(cfg), make_declare_position(cfg),
        make_power_weighted_vote(PowerWeightedVoteConfig()),
        make_tax_and_redistribute(cfg), make_update_regime(cfg),
        make_rewire_delegation(cfg),
    ]
    steps.extend(mechanism_transforms)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: DelegativePolityConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: write engagement (action space [0,∞)),
    then run the compiled pipeline. The substrate is deterministic given the
    init — ``rng_key`` is still threaded for mechanisms that may want it."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        state = state.update_node_attrs("engagement", jnp.maximum(actions, 0.0))
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Raw per-step readouts; the delegation matrix is (N, N) and evolves — read
    its final form from ``finals.adj_matrices``, not the trace (its dominant
    edges ship as the O(N) ``top_delegate_target`` index series). ``ideal`` is
    static but traced so metrics can compute the citizen median per seed."""
    return {
        "influence": state.node_attrs["influence"],
        "wealth": state.node_attrs["wealth"],
        "ideal": state.node_attrs["ideal"],
        "top_delegate_target": top_target(state.adj_matrices["delegation"]),
        "policy_target": state.global_attrs["policy_target"],
        "enforcement": state.global_attrs["enforcement"],
        "redelegation_friction": state.global_attrs["redelegation_friction"],
    }
