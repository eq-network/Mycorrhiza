"""
Substrate dynamics for Value Contagion — two-sided competing contagion, OPEN at
the agent boundary.

One synchronous tick: every agent may adopt the *other* cultural variant with
probability ``1 − Π_j (1 − β_j)`` over neighbors j currently holding it, where
``β_j = beta · p_advantage^{c_j} · effort_j`` — the advantage rides on the
VARIANT (AI-origin), not the carrier, so a converted human spreads AI-origin
culture at full strength. Three deliberate asymmetries:

- **AI nodes are a frozen reservoir** (never flip): AI keeps generating its own
  culture whatever humans do.
- **Native reversion**: humans holding AI-origin culture also revert at rate
  ``recovery``, independent of neighbors — without it the reservoir makes
  "everyone AI-cultured" the only fixed point and the pluralism regime cannot
  exist (see config.py).
- **Complex-contagion gate**: adoption requires ≥ ``k_threshold`` exposures
  (Centola 2010); at k=1 this is ordinary independent-cascade contagion.

Two-sidedness is what stabilizes the parallel-cultures corner: a stray convert
inside a cohesive cluster is flipped back by majority pressure. The decision
rule is NOT here — the boundary exposes per-agent broadcast effort (how loudly
you transmit as a source), written by ``step_fn`` from the actions; the catalog
default ``BroadcastPolicy(1.0)`` recovers the classic count rule exactly.
Randomness threads through ``global_attrs["rng_key"]`` (governed_commons
pattern); ``step_fn`` seeds it from its key argument.

Every use of ``friendship`` here is a matvec (``W @ v``), which is why
``cfg.sparse_friendship`` is a drop-in: ``@`` dispatches identically on a dense
array and a BCOO. The one exception is the degree vector in ``observe_fn`` —
see ``networks.row_sums``.
"""
from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.category import Transform, transform
from cilib.core.pipeline import compile_pipeline

from ..networks import row_sums
from .config import ValueContagionConfig


def _split_key(state: GraphState):
    """Consume and advance the RNG key in global_attrs (governed_commons pattern)."""
    key = state.global_attrs["rng_key"]
    key, sub = jr.split(key)
    return state.update_global_attr("rng_key", key), sub


# --- the boundary: what an agent observes --------------------------------------

def observe_fn(state: GraphState):
    """Per-agent observation: ``[own_culture, local_ai_exposure_share]`` — what
    you hold and how much AI-origin culture your friends are showing you.
    Shape (N, 2)."""
    W = state.adj_matrices["friendship"]
    c = state.node_attrs["culture"]
    # row_sums, not jnp.sum: the latter raises on a sparse `friendship`
    exposure = (W @ c) / jnp.maximum(row_sums(W), 1.0)
    return jnp.stack([c, exposure], axis=1)


# --- adoption: the whole substrate in one transform ------------------------------

def make_adopt(cfg: ValueContagionConfig):
    """Synchronous competing-contagion update (module docstring has the rule)."""

    @transform(reads=["friendship", "culture", "broadcast_effort", "rng_key"],
               writes=["culture", "rng_key"])
    def adopt(state: GraphState) -> GraphState:
        state, sub = _split_key(state)
        W = state.adj_matrices["friendship"]
        c = state.node_attrs["culture"]
        effort = state.node_attrs["broadcast_effort"]

        # per-SOURCE transmissibility; AI-origin variants carry the advantage
        beta_src = jnp.clip(
            cfg.beta * jnp.where(c == 1.0, cfg.p_advantage, 1.0) * effort,
            0.0, 1.0 - 1e-6)
        log_keep = jnp.log1p(-beta_src)                # log(1 - β_j), 0 for silent

        m_ai = W @ c                                   # exposure counts (the k gate)
        m_human = W @ (1.0 - c)
        p_fwd = jnp.where(m_ai >= cfg.k_threshold,
                          1.0 - jnp.exp(W @ (log_keep * c)), 0.0)
        p_bwd = jnp.where(m_human >= cfg.k_threshold,
                          1.0 - jnp.exp(W @ (log_keep * (1.0 - c))), 0.0)
        p_bwd = 1.0 - (1.0 - p_bwd) * (1.0 - cfg.recovery)   # + native reversion

        flip = jr.bernoulli(sub, jnp.where(c == 0.0, p_fwd, p_bwd))
        new_c = jnp.where(flip, 1.0 - c, c)
        new_c = jnp.where(state.node_types == 1, c, new_c)   # frozen AI reservoir
        return state.update_node_attrs("culture", new_c)
    return adopt


# --- bookkeeping ----------------------------------------------------------------

def make_step_counter(cfg: ValueContagionConfig):
    @transform(reads=["step"], writes=["step"])
    def step_counter(state: GraphState) -> GraphState:
        return state.update_global_attr("step", state.global_attrs["step"] + 1)
    return step_counter


# --- composition + trace ---------------------------------------------------------

def build_steps(cfg: ValueContagionConfig,
                mechanism_transforms: Sequence[Transform] = ()) -> List[Transform]:
    """The post-action pipeline in program order: substrate, mechanisms, counter.
    The mechanisms slot is where curation/provenance defenses attach later."""
    steps: List[Transform] = [make_adopt(cfg)]
    steps.extend(mechanism_transforms)
    steps.append(make_step_counter(cfg))
    return steps


def build_step_fn(cfg: ValueContagionConfig,
                  mechanism_transforms: Sequence[Transform] = ()):
    """``(state, actions, key) -> state``: write the broadcast efforts (the game
    enforces its [0, 1] action space), then run the compiled pipeline."""
    pipeline = compile_pipeline(build_steps(cfg, mechanism_transforms))

    def step_fn(state: GraphState, actions, key) -> GraphState:
        state = state.update_global_attr("rng_key", key)
        state = state.update_node_attrs("broadcast_effort",
                                        jnp.clip(actions, 0.0, 1.0))
        return pipeline(state)
    return step_fn


def default_trace(state: GraphState):
    """Raw per-step readouts; reduction happens in metrics.py / the example.
    The friendship network is static per run — read it from finals, not here."""
    return {
        "culture": state.node_attrs["culture"],
        "broadcast_effort": state.node_attrs["broadcast_effort"],
    }
