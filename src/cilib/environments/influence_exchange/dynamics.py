"""
Substrate dynamics for Influence Exchange — DeGroot with an endogenous
listening matrix, OPEN at the agent boundary.

Round (post-action pipeline, composed via ``compile_pipeline``):

    amplify -> rewire -> influence_update -> opinion_update -> [mechanisms] -> counter

- **amplify**: pure function of the clock — AI nodes' attractiveness multiplier
  is ``amplification`` from ``amp_onset``, 1 before (the compute_economy
  arrival idiom: the threat is a schedule, not a decision).
- **rewire** (preferential attachment): each citizen shifts ``update_rate`` of
  their off-diagonal listening mass toward the attractiveness distribution
  ``a_j ∝ (influence_j + eps)^gamma · amplification_j · cap_scale_j ·
  attract_boost_j · engagement_j``. The diagonal (self_weight) is fixed; AI
  rows are frozen — AI power is being listened TO. Deterministic: the only
  stochasticity in this environment is the initial draw.
- **influence_update**: one power-iteration step ``v <- normalize(W^T v)`` —
  for frozen W this converges to the left eigenvector, i.e. the DeGroot
  consensus weights (Golub-Jackson); in-loop it makes influence a measurable,
  evolving field rather than offline math.
- **opinion_update**: ``x <- Wx`` with AI opinions pinned at their bias (the
  value_contagion frozen-reservoir idiom).

The mechanisms slot sits after the substrate: a defense reads this round's
influence and rewrites next round's listening/attractiveness — same "this
round's outcome sets next round's rules" convention as governed_commons.

The boundary: agents observe ``[influence, opinion]`` and choose ``engagement``
(how loudly they compete for attention); the catalog's constant-effort
``BroadcastPolicy`` closes it (political engagement = broadcast effort — the
same boundary as value_contagion, deliberately).
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from .config import InfluenceExchangeConfig


# --- the boundary: what an agent observes --------------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[own_influence, own_opinion]``. Shape (N, 2)."""
    return jnp.stack([state.node_attrs["influence"],
                      state.node_attrs["opinion"]], axis=1)


# --- scheduled algorithmic amplification -----------------------------------------

def make_amplify(cfg: InfluenceExchangeConfig):
    @transform(reads=["step"], writes=["amplification"])
    def amplify(state: GraphState) -> GraphState:
        on = state.global_attrs["step"] >= cfg.amp_onset
        amp = jnp.where((state.node_types == 1) & on, cfg.amplification, 1.0)
        return state.update_node_attrs("amplification", amp)
    return amplify


# --- preferential-attachment rewiring of the listening matrix --------------------

def make_rewire(cfg: InfluenceExchangeConfig):
    @transform(reads=["listening", "influence", "amplification", "cap_scale",
                      "attract_boost", "engagement"],
               writes=["listening"])
    def rewire(state: GraphState) -> GraphState:
        W = state.adj_matrices["listening"]
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

        offdiag = W * (1.0 - eye)
        offdiag = offdiag / jnp.maximum(jnp.sum(offdiag, axis=1, keepdims=True), 1e-12)
        # a row with nowhere to drift (all-silent world) keeps its listening —
        # otherwise the zero target would silently drain row-stochasticity
        mixed = jnp.where(t_mass > 1e-9,
                          (1.0 - cfg.update_rate) * offdiag + cfg.update_rate * target,
                          offdiag)
        W_new = cfg.self_weight * eye + (1.0 - cfg.self_weight) * mixed

        is_ai_row = (state.node_types == 1)[:, None]
        return state.update_adj_matrix("listening", jnp.where(is_ai_row, W, W_new))
    return rewire


# --- influence: one power-iteration step toward the left eigenvector -------------

def make_influence_update(cfg: InfluenceExchangeConfig):
    @transform(reads=["listening", "influence"], writes=["influence"])
    def influence_update(state: GraphState) -> GraphState:
        v = state.adj_matrices["listening"].T @ state.node_attrs["influence"]
        return state.update_node_attrs("influence", v / jnp.maximum(jnp.sum(v), 1e-12))
    return influence_update


# --- opinion pooling: Friedkin-Johnsen anchored DeGroot, AI reservoir pinned -----

def make_opinion_update(cfg: InfluenceExchangeConfig):
    """``x <- (1-λ)·signal + λ·(Wx)`` for citizens (Friedkin-Johnsen 1990), AI
    pinned at its bias. The anchor is load-bearing — see config.py."""

    @transform(reads=["listening", "opinion", "signal"], writes=["opinion"])
    def opinion_update(state: GraphState) -> GraphState:
        x = state.node_attrs["opinion"]
        pooled = ((1.0 - cfg.susceptibility) * state.node_attrs["signal"]
                  + cfg.susceptibility * (state.adj_matrices["listening"] @ x))
        return state.update_node_attrs(
            "opinion", jnp.where(state.node_types == 1, x, pooled))
    return opinion_update


# --- bookkeeping ----------------------------------------------------------------

def make_step_counter(cfg: InfluenceExchangeConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace ---------------------------------------------------------

def build_steps(cfg: InfluenceExchangeConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """Post-action pipeline in program order: substrate, mechanisms, counter."""
    steps: List[Transform] = [
        make_amplify(cfg), make_rewire(cfg),
        make_influence_update(cfg), make_opinion_update(cfg),
    ]
    steps.extend(mechanism_transforms)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: InfluenceExchangeConfig,
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
    """Raw per-step readouts; the listening matrix is (N, N) and evolves — read
    its final form from ``finals.adj_matrices``, not the trace."""
    return {
        "opinion": state.node_attrs["opinion"],
        "influence": state.node_attrs["influence"],
        "amplification": state.node_attrs["amplification"],
        "cap_scale": state.node_attrs["cap_scale"],
    }
