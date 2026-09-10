"""
Economy family — ALLOCATE (the GD game's three-family design).

Build spec: the 2026-07-31 economy spec; measured lever effects and the constraints:
``docs/gd-game-three-families.md``.

The verb is *allocate*: each tick the town's budget is split over
[consume, invest, broadcast, lobby, save], and the four spend bands are the
whole game — **save is not a slider, it is the remainder**. ``make_allocate``
in ``ledger_society/dynamics.py`` already is the mechanic
(``spends = alloc[:, :4] * budget``; column 4 is never spent, it stays in
``wealth``), so this family adds no dynamics — it *steers the field the
kernel already reads*.

**Which field takes effect.** ``allocate`` reads ``allocation``, but
``build_step_fn`` overwrites ``allocation`` from the policy's actions at the
head of every tick, so anything this transform wrote there would be discarded
before ``allocate`` ever saw it. ``observe_fn`` returns ``alloc_pref`` and the
closing ``SpendSharePolicy`` passes it through, so **``alloc_pref`` is the
field that takes effect** — exactly the field the attention-campaign card
already writes. Levers run in the mechanism slot, after ``allocate``, so a
change lands **the following tick**. The receipt must say "from next year";
do not hide the lag (spec §"The controls").

**Plan semantics: deltas, not absolutes.** The plan row carries four *deltas*
applied to the human rows of ``alloc_pref``, then each resulting band is
clipped to its declared range. A zero row is ``pref + 0.0`` — bit-identical,
the sealing convention this repo already tests. AI rows are never touched.

**Conservation without silent renormalisation.** The residual (column 4)
absorbs exactly what the bands took: ``save -= sum(bands_new) - sum(bands_old)``,
so the row's sum is *preserved*, not rescaled. The spec is explicit that the
panel must not silently renormalise — a player who drags one slider must not
watch three others move. The only rescaling here is the infeasibility guard:
if the clipped bands would claim more than the whole row, they are scaled back
to the row's sum (the second line of defence behind the panel's slider lock).
At any feasible row the guard multiplies by exactly 1.0.

Plan column order is ``ECONOMY_LEVERS``. The array is DATA — a dynamic pytree
child of ``global_attrs`` — so one compiled program serves every plan
(remote-engine R0 idiom, ``make_policy_levers``). ``ledger_society``'s state
factory does not create ``economy_plan``; attach it with
``attach_economy_plan`` before the run.

Not composable with ``make_policy_levers`` or ``make_interventions`` in the
same pipeline — all three write ``wealth`` and the allocation channel; the
card game, the four-lever policy game and the three-family game are
alternative closures.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform

# Plan column order. Four band deltas + the conserving AI->human levy.
ECONOMY_LEVERS = ("d_consume", "d_invest", "d_broadcast", "d_lobby", "levy_rate")
ECONOMY_PLAN = "economy_plan"


@dataclasses.dataclass(frozen=True)
class EconomyLeverConfig:
    """Ranges are the server's whitelist AND the in-transform clip.

    Parameter typing (house rule; every field probed or sourced 2026-07-31):

    - ``band_lo`` / ``band_hi`` — **arbitrary-but-swept**. The spec's table:
      consume 0.30-0.85, invest 0.00-0.40, broadcast 0.00-0.30, lobby
      0.00-0.20. The neutral points they bracket are ``LedgerSocietyConfig.
      human_alloc`` = (0.70, 0.05, 0.03, 0.02, 0.20), already typed
      tuned-for-legibility. **Invest above 0.25 is beyond the probe**: the
      three-families deposit measured invest 0.05 -> 0.25 (6 seeds, one field
      at a time, no CIs) and nothing above it. 0.40 is a designer's range, not
      a measured one — sweep before quoting any number from up there.
    - ``levy_max`` — **tuned-for-legibility**, carried over unchanged from
      ``PolicyLeverConfig.levy_max`` so the two closures price the levy alike.

    The measured ordering claims this family rests on (deposit, 2026-07-31,
    harbor dials, wait path, 6 seeds, medians, **no CIs** — ordering only):
    invest 0.05 -> 0.25 is the largest lever in the model and the only one
    that substantially delays the flip; broadcast is near-inert on the money
    ledger; drawdown is the one negative-signed economy lever probed.
    """

    # order matches ECONOMY_LEVERS[:4]
    band_lo: tuple = (0.30, 0.00, 0.00, 0.00)
    band_hi: tuple = (0.85, 0.40, 0.30, 0.20)
    levy_max: float = 0.3
    eps: float = 1e-12          # guard divisor; never reached at feasible rows


def neutral_economy_plan(horizon: int) -> jnp.ndarray:
    """The sealed plan: ``(horizon, 5)`` zeros. Every lever is exactly neutral
    at 0, so a run carrying this plan is bit-identical to one with no economy
    transform at all (asserted in the tests)."""
    return jnp.zeros((int(horizon), len(ECONOMY_LEVERS)), dtype=jnp.float32)


def attach_economy_plan(state: GraphState, plan) -> GraphState:
    """Put the plan on the state as a dynamic pytree child. Values are data —
    two plans of the same horizon share one compiled program."""
    return state.update_global_attr(
        ECONOMY_PLAN, jnp.asarray(plan, dtype=jnp.float32))


def make_economy_levers(cfg: EconomyLeverConfig = EconomyLeverConfig()):
    """Per-tick allocation-plan reader for the live game.

    Writes ``alloc_pref`` (the human rows) and ``wealth`` (the levy). Reads and
    writes nothing the culture family (attention-kernel shape) or the politics
    family (enforcement stock) owns.
    """
    lo = jnp.asarray(cfg.band_lo, dtype=jnp.float32)[None, :]
    hi = jnp.asarray(cfg.band_hi, dtype=jnp.float32)[None, :]

    @transform(reads=["step", ECONOMY_PLAN, "alloc_pref", "wealth"],
               writes=["alloc_pref", "wealth"])
    def economy_levers(state: GraphState) -> GraphState:
        plan = state.global_attrs[ECONOMY_PLAN]
        row = plan[jnp.clip(state.global_attrs["step"], 0, plan.shape[0] - 1)]

        is_human = state.node_types == 0
        h = is_human.astype(jnp.float32)
        n_h = jnp.maximum(jnp.sum(h), 1.0)

        # --- the four bands: delta, clip to range, residual absorbs the move ---
        pref = state.node_attrs["alloc_pref"]
        bands0 = pref[:, :4]
        save0 = pref[:, 4]
        budget0 = jnp.sum(pref, axis=1, keepdims=True)     # the row's own sum

        bands = jnp.clip(bands0 + row[None, :4], lo, hi)
        # infeasibility guard: the four bands can claim at most the whole row.
        # At any feasible row this is min(1.0, >1) = 1.0 exactly — no rescaling,
        # which is what the spec's no-silent-renormalisation rule demands.
        s = jnp.sum(bands, axis=1, keepdims=True)
        bands = bands * jnp.minimum(1.0, budget0 / jnp.maximum(s, cfg.eps))
        moved = jnp.sum(bands, axis=1) - jnp.sum(bands0, axis=1)
        save = jnp.maximum(save0 - moved, 0.0)
        pref_out = jnp.where(is_human[:, None],
                             jnp.concatenate([bands, save[:, None]], axis=1),
                             pref)

        # --- the levy: conserving AI -> human transfer (the built card) --------
        levy = jnp.clip(row[4], 0.0, cfg.levy_max)
        wealth = state.node_attrs["wealth"]
        take = levy * wealth * (1.0 - h)
        wealth = wealth - take + jnp.sum(take) / n_h * h

        state = state.update_node_attrs("alloc_pref", pref_out)
        return state.update_node_attrs("wealth", wealth)
    return economy_levers
