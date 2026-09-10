"""
Culture family — SHAPE. The kernel-shaping levers of the GD game's three-family
design, built to the 2026-07-31 culture spec; measured lever effects and the
constraints: docs/gd-game-three-families.md.

The verb is *shape*: you cannot move attention directly, only the parameters
governing how it moves, and they act with a lag. The attention ledger's time
constant is ``1 / update_rate_w`` — 12.5 ticks at the default 0.08, 33 ticks at
the slow end of the declared range. That number is derived from the update rule
in ``environments/attachment.py``, not measured.

**The plan is data.** One ``(T, 3)`` float array in
``global_attrs["culture_plan"]``, a dynamic pytree child read by row per tick,
so one compiled program serves every plan. Column order is ``CULTURE_LEVERS``.
The seam this family needs from the substrate already exists: ``state.py``
promoted ``gamma_w_now`` and ``update_rate_w_now`` to per-tick globals seeded
from their config fields, and ``dynamics.make_rewire_listening`` reads them;
``reach_cut_now`` predates this family. Writing a global rather than a node
attribute is what keeps this family's writes disjoint from economy's and
politics'.

**Neutrality.** Both shape levers are carried as DELTAS neutral at 0, not raw
values, so the sealing convention holds and ``gamma_w = 1.0`` stays an anchor in
the config instead of being smuggled into the plan array. At a zero row the
transform writes back exactly the config-seeded values — float32 ``base + 0.0``
is exact and both clips are inactive — so a neutral plan is bit-identical to no
plan at all, downstream matrix included. That is asserted as a test.

**What the probe says these levers do.** Harbor dials, wait path, 6 seeds, one
field at a time, journey-window metrics against the unchanged baseline; medians,
no CIs, so these are ordering claims and the magnitudes travel only with this
sentence. ``gamma_w`` 1.0 -> 0.6 moved the composite +0.023 and ``gamma_w``
1.0 -> 1.4 moved it -0.010; ``update_rate_w`` 0.08 -> 0.03 moved the composite
+0.013 and moved the flip tick from 26 to 36. This family is weak on the score
and is the only one that touches *when* the town turns. Speeding the town up is
unprobed. Every effect above was measured with the field changed from t=0, so a
*late* culture lever has never been measured and "near-worthless after the flip"
is a design conjecture, not a measurement.

**Two things the spec asks for that this file does not do.**

1. *The town's voice* — the broadcast column of ``alloc_pref`` — is a node
   attribute of the conserved allocation vector, which the economy family owns
   and writes. Implementing it here would break the disjoint-writes invariant.
   It belongs on the economy screen; culture reads its consequences.
2. *Enforcement upkeep for holding a lever off default* has no field this family
   may write. ``enforcement`` is the politics family's stock. The cost model is
   therefore exported as ``culture_upkeep``, a pure per-tick charge in
   enforcement units that whoever owns ``enforcement`` debits, and it is charged
   nowhere in this module. The cost split was flagged as crossing the brief;
   this leaves the coefficient priced and the ruling open.

Parameter typing (house rule), probed 2026-07-31 where noted.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform

# plan column order: two deltas neutral at 0, one fraction neutral at 0
CULTURE_LEVERS = ("gamma_w_delta", "update_rate_w_delta", "reach_cut")
CULTURE_PLAN = "culture_plan"        # the global_attrs key the transform reads


@dataclasses.dataclass(frozen=True)
class CultureLeverConfig:
    """Bases, ranges and the upkeep coefficient for the culture plan.

    The bases mirror ``LedgerSocietyConfig``; use ``from_society`` to bind them
    to an actual config rather than trusting the copy. Ranges are the server's
    whitelist AND the in-transform clip.
    """
    # bases the deltas are measured from
    gamma_w_base: float = 1.0            # anchored: linear attachment is
                                         # share-neutral (Krapivsky-Redner)
    update_rate_w_base: float = 0.08     # anchored to influence_exchange's default
    # declared delta ranges (arbitrary-but-swept; the spec's table, 2026-07-31)
    gamma_w_delta_max: float = 0.4       # gamma in [0.6, 1.4]
    update_rate_w_delta_lo: float = -0.05    # rate floor 0.03: time constant 33
    update_rate_w_delta_hi: float = 0.07     # rate ceiling 0.15: time constant 6.7
    reach_cut_max: float = 1.0           # the built cap, unchanged
    # the cost of holding the square off its own shape, in enforcement units per
    # tick per unit of summed normalized lever intensity. Charged by whoever owns
    # enforcement, never here (tuned-for-legibility; mirrors
    # PolicyLeverConfig.upkeep, which is the same coefficient for the same job)
    upkeep: float = 0.004

    @classmethod
    def from_society(cls, society_cfg, **overrides) -> "CultureLeverConfig":
        """Bind the bases to a substrate config by attribute, no import."""
        return cls(gamma_w_base=float(society_cfg.gamma_w),
                   update_rate_w_base=float(society_cfg.update_rate_w),
                   **overrides)


def neutral_culture_plan(horizon: int):
    """The sealed plan: every tick at the config's own shape."""
    return jnp.zeros((horizon, len(CULTURE_LEVERS)), dtype=jnp.float32)


def attach_culture_plan(state: GraphState, plan) -> GraphState:
    """Put the plan on the state as a dynamic pytree child. Values are data —
    two plans of the same horizon share one compiled program. ``state.py`` does
    not create this key, so it is attached here or by the runner before t=0."""
    return state.update_global_attr(
        CULTURE_PLAN, jnp.asarray(plan, dtype=jnp.float32))


def culture_upkeep(row, cfg: CultureLeverConfig = CultureLeverConfig()):
    """Enforcement drain this plan row asks for, as a traced scalar.

    Proportional to summed normalized lever intensity, so it is exactly 0 at a
    neutral row and symmetric in the sign of a delta — flattening the square
    costs what sharpening it costs. Pure: this module never applies it, because
    ``enforcement`` is the politics family's write field. Compose it there.
    """
    row = jnp.asarray(row)
    intensity = (jnp.abs(jnp.clip(row[0], -cfg.gamma_w_delta_max,
                                  cfg.gamma_w_delta_max)) / cfg.gamma_w_delta_max
                 + jnp.abs(jnp.clip(row[1], cfg.update_rate_w_delta_lo,
                                    cfg.update_rate_w_delta_hi))
                 / max(abs(cfg.update_rate_w_delta_lo),
                       abs(cfg.update_rate_w_delta_hi))
                 + jnp.clip(row[2], 0.0, cfg.reach_cut_max) / cfg.reach_cut_max)
    return cfg.upkeep * intensity


def make_culture_levers(cfg: CultureLeverConfig = CultureLeverConfig()):
    """The culture family's whole plan as one transform for the mechanism slot.

    Reads ``global_attrs["culture_plan"]``, shape ``(T, 3)``; writes only the
    three substrate globals the attention path consumes next tick. Not
    composable with ``make_policy_levers`` in the same pipeline — both write
    ``reach_cut_now``; the four-lever policy game and the three families are
    alternative closures.
    """

    @transform(reads=["step", "culture_plan"],
               writes=["gamma_w_now", "update_rate_w_now", "reach_cut_now"])
    def culture_levers(state: GraphState) -> GraphState:
        plan = state.global_attrs[CULTURE_PLAN]
        row = plan[jnp.clip(state.global_attrs["step"], 0, plan.shape[0] - 1)]
        # clipped in-transform as a second line of defence behind the server's
        # range whitelist; every clip is inactive at a neutral row
        g_d = jnp.clip(row[0], -cfg.gamma_w_delta_max, cfg.gamma_w_delta_max)
        r_d = jnp.clip(row[1], cfg.update_rate_w_delta_lo,
                       cfg.update_rate_w_delta_hi)
        cut = jnp.clip(row[2], 0.0, cfg.reach_cut_max)
        # physical bounds on the resulting substrate values: a negative exponent
        # would invert attractiveness, and the kernel needs a rate in [0, 1]
        gamma = jnp.maximum(cfg.gamma_w_base + g_d, 0.0)
        rate = jnp.clip(cfg.update_rate_w_base + r_d, 0.0, 1.0)

        state = state.update_global_attr("gamma_w_now", gamma)
        state = state.update_global_attr("update_rate_w_now", rate)
        return state.update_global_attr("reach_cut_now", cut)
    return culture_levers
