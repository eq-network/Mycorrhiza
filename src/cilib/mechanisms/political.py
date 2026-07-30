"""
Political family — attention-structure defenses for influence exchange.

Both entries treat the listening matrix as the constitution: who gets heard is
the rule being governed. Two composable defenses (Gradual Disempowerment §4's
delegative/sortition family):

- ``sortition``: a civic lottery for attention — on its schedule tick, every
  citizen's off-self listening is blended ``share`` of the way back toward the
  uniform distribution over citizens. Attention captured by any node (AI or
  human hub) is returned to the demos; AI rows are untouched (their listening
  was never the problem). Timing belongs to the schedule: wrap with
  ``core.schedule.scheduled(make_sortition(cfg), cadence=25)``.
- ``influence_cap``: an attractiveness damper, not a rewiring — any node whose
  influence share exceeds ``cap_share`` gets ``cap_scale = cap_share/share``
  (< 1), which the substrate's preferential-attachment rewire multiplies into
  its attractiveness. Over-cap nodes stop *attracting* new attention; existing
  listening decays away organically. Generic: reads only ``influence``.

Family contract: disjoint writes ({listening} vs {cap_scale}), so
``compile_pipeline`` composes them in any order without conflict — same
convention as the democracy family.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform


@dataclasses.dataclass(frozen=True)
class SortitionConfig:
    share: float = 0.5    # fraction of off-self listening returned to the demos
                          # (calibrated 2026-07-27: with cap 0.04 + cadence 15
                          # the four A4 conditions separate at T=400)
    adj_key: str = "listening"   # the attention adjacency the lottery acts on
                                 # ("listening" in influence_exchange, "delegation"
                                 # in delegative_polity — same constitution idiom)


def make_sortition(cfg: SortitionConfig):
    """Blend citizens' listening ``share`` of the way toward uniform-over-citizens.

    The citizen mask comes from ``state.node_types`` (type 0), so the mechanism
    is environment-size agnostic. Row-stochasticity is preserved exactly: the
    blend target is itself a distribution over each row.
    """

    @transform(reads=[cfg.adj_key], writes=[cfg.adj_key])
    def sortition(state: GraphState) -> GraphState:
        W = state.adj_matrices[cfg.adj_key]
        N = W.shape[0]
        eye = jnp.eye(N)
        is_citizen_col = (state.node_types == 0).astype(W.dtype)[None, :]

        # uniform over citizens, excluding self, per row
        lottery = is_citizen_col * (1.0 - eye)
        lottery = lottery / jnp.maximum(jnp.sum(lottery, axis=1, keepdims=True), 1e-12)

        diag = jnp.diag(jnp.diag(W))                     # keep each row's self-anchor
        off = W - diag
        off_mass = jnp.sum(off, axis=1, keepdims=True)
        blended = (1.0 - cfg.share) * off + cfg.share * off_mass * lottery

        W_new = diag + blended
        is_ai_row = (state.node_types == 1)[:, None]
        return state.update_adj_matrix(cfg.adj_key, jnp.where(is_ai_row, W, W_new))
    return sortition


@dataclasses.dataclass(frozen=True)
class InfluenceCapConfig:
    cap_share: float = 0.04   # max influence share before attraction damping
                              # (~1.4x an equal split at N=34; see SortitionConfig)


def make_influence_cap(cfg: InfluenceCapConfig):
    """Damp the attractiveness of any node above the cap: ``cap_scale =
    min(1, cap_share / influence_share)``. The substrate's rewire multiplies
    ``cap_scale`` into attractiveness, so the cap acts on attention *growth*."""

    @transform(reads=["influence"], writes=["cap_scale"])
    def influence_cap(state: GraphState) -> GraphState:
        v = state.node_attrs["influence"]
        share = v / jnp.maximum(jnp.sum(v), 1e-12)
        scale = jnp.minimum(1.0, cfg.cap_share / jnp.maximum(share, 1e-12))
        return state.update_node_attrs("cap_scale", scale)
    return influence_cap
