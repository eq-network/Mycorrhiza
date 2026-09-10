"""
Politics family — SPEND (the GD game's three-family design).

Build spec: ``round-table/gd-game-dynamics-2026-07-31/specs/politics.md``
(Marcus Oyelaran, 2026-07-31); measured lever effects and the constraints:
``docs/gd-game-three-families.md``.

The verb is *spend*: ``enforcement`` is a bounded stock in [0, 1] that does two
jobs in the substrate — it multiplies the tax the town actually collects
(``rate = policy_target x enforcement`` in ``tax_and_redistribute``) and it
gates the ballot ledger's freedom to re-draw (``churn_eff = churn_now x
redelegation_friction`` in ``rewire_delegation``, and friction *is* enforcement).
Every lever here moves that number, and it refills only through repair.

**This family owns ``enforcement``, so it pays the whole town's political
bill.** The economy family's levy and the culture family's reach cap are
political acts billed to the same stock, but neither of those modules may write
it (disjoint writes). Their charge therefore arrives here as data, in plan
column ``external_intensity``, in the same normalized units the spec's total
uses::

    intensity = levy/0.3 + cut/1.0 + sortition/0.2      (spec, "The controls")

``external_intensity`` carries the first two terms, ``sortition_rate`` supplies
the third, and ``cfg.upkeep`` — the coefficient from ``PolicyLeverConfig``,
reused rather than reinvented — converts the sum into enforcement per tick.
``external_intensity_of`` builds that number from the sibling families' plan
rows, and ``culture.culture_upkeep`` is the sibling's own pricing function, used
as its author asked it to be used. Politics is the only tab showing a total, and
the only transform that debits the stock.

**The plan is data.** One ``(T, 5)`` float array in
``global_attrs["politics_plan"]``, a dynamic pytree child read by row per tick,
so one compiled program serves every plan (remote-engine R0 idiom,
``make_policy_levers``). Column order is ``POLITICS_LEVERS``.
``ledger_society``'s state factory does not create ``politics_plan``; attach it
with ``attach_politics_plan`` before the run.

**Neutrality: every column is neutral at exactly 0, and the two substrate rates
read 0 as "hold".** The spend and sortition columns are rates, neutral at 0 in
the ordinary way. ``repair_rate`` and ``entrenchment_gain`` are *absolute*
substrate values whose neutral point is whatever the config seeded into
``repair_rate_now`` / ``entrenchment_gain_now``, so a 0 in those columns writes
back the value already on the state — bit-identical for any config, not just the
default one. The cost is that the plan cannot force either rate to exactly zero;
a pure-ratchet scenario sets ``LedgerSocietyConfig.repair_rate = 0`` and leaves
the column neutral. A neutral plan is bit-identical to no politics transform at
all, delegation matrix included, in the eager tier AND under ``lax.scan``; both
are asserted as tests. The scan rung is the load-bearing one — see the sortition
block for why the ballot write is ``D + sort_r x delta`` and not the
algebraically equal decompose-and-recompose ``make_policy_levers`` uses.

**Two live controls, two scenario dials.** The spec ships exactly two player
sliders — fund the office and seats by lot — and rules the other two columns out
of the player's hands with reasons this file does not get to overrule:
``repair_rate`` is the strongest politics lever in the probe (0.02 -> 0.04,
+0.206 enforcement) and it is *free*, and a free dial is not a decision;
``entrenchment_gain`` stays 0.0 in the shipped scenario because whether ballot
concentration erodes enforcement by itself is a claim about the world that owes
a dated ASSUMPTIONS entry. Both columns exist because the family is the declared
writer of those two globals and the variants need them, not because the game
exposes them.

**What the probe says these levers do.** Harbor dials, wait path, 6 seeds, one
field at a time, journey-window metrics against the unchanged baseline; medians,
**no CIs**, so these are ordering claims and the magnitudes travel only with
this sentence. Config ``repair_rate`` 0.02 -> 0.04 moved enforcement +0.206 and
wealth +0.132; ``entrenchment_gain`` 0 -> 0.02 moved wealth -0.007 and is the
only clean downside lever found; three ballot-shape fields (``churn``,
``self_weight_d``, ``attention_to_ballots``) land within 0.005 of each other on
every readout, which is why this family ships one ballot lever and not three.
Nothing in the probe measured the *spend* column or ``sortition_rate``: the
sortition blend and the ``repair_efficiency x drip_pc x (1 - e)`` uplift are
arithmetic off ``dynamics.update_regime`` and ``make_policy_levers``, not
measurements, and the spec says so of its own price line.

Not composable with ``make_policy_levers`` or ``make_interventions`` in the same
pipeline — all three write ``enforcement``, ``wealth`` and ``delegation``; the
card game, the four-lever policy game and the three-family game are alternative
closures. Within the three families the only shared write is ``wealth``, which
economy moves by the conserving levy and politics drains into the declared
``intervention_spend`` sink; that overlap is physical, not a naming accident —
the office is funded out of the same household hoard that funds investment, and
the spec's third failure mode is exactly that trade.

Parameter typing (house rule), probed or sourced 2026-07-31 where noted.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.category import transform

from ..interventions import PolicyLeverConfig

# Plan column order. Two live player controls, two scenario dials, one inbound
# bill from the sibling families.
POLITICS_LEVERS = ("repair_spend_rate", "sortition_rate", "repair_rate",
                   "entrenchment_gain", "external_intensity")
POLITICS_PLAN = "politics_plan"


@dataclasses.dataclass(frozen=True)
class PoliticsLeverConfig:
    """Ranges are the server's whitelist AND the in-transform clip.

    Parameter typing (house rule; 2026-07-31):

    - ``repair_spend_max`` — **tuned-for-legibility**, carried over unchanged
      from ``PolicyLeverConfig.repair_max`` so the four-lever closure and this
      one price the office alike. It is a fraction of every household's hoard
      per tick; at the top setting the drip is 5% of savings per tick, which the
      spec names as a losing trade against invest (+0.539 wealth, the model's
      strongest lever) and keeps in the game for that reason.
    - ``sortition_max`` — **tuned-for-legibility**, from
      ``PolicyLeverConfig.sortition_max``.
    - ``repair_rate_max`` — **arbitrary-but-swept**: the probe swept the config
      field 0.02 -> 0.04; 0.05 is the declared ceiling, above the probed range,
      so no number from up there travels without its own sweep.
    - ``entrenchment_gain_max`` — **arbitrary-but-swept**: the probe measured
      0 -> 0.02 only.
    - ``external_intensity_max`` — **arbitrary-but-swept**: 4.0 bounds the
      siblings' worst case, one normalized unit for the levy plus up to three
      for ``culture_upkeep``'s three terms.
    - ``repair_efficiency``, ``upkeep`` — **tuned-for-legibility**, both taken
      from ``PolicyLeverConfig`` unchanged. ``upkeep`` is the same coefficient
      the culture family prices with, for the same job.
    """

    repair_spend_max: float = PolicyLeverConfig.repair_max          # 0.05
    sortition_max: float = PolicyLeverConfig.sortition_max          # 0.20
    repair_rate_max: float = 0.05
    entrenchment_gain_max: float = 0.05
    external_intensity_max: float = 4.0
    repair_efficiency: float = PolicyLeverConfig.repair_efficiency  # 0.5
    upkeep: float = PolicyLeverConfig.upkeep                        # 0.004


def neutral_politics_plan(horizon: int) -> jnp.ndarray:
    """The sealed plan: ``(horizon, 5)`` zeros. Every column is exactly neutral
    at 0 — the two rate columns read 0 as "hold the substrate's value" — so a
    run carrying this plan is bit-identical to one with no politics transform at
    all (asserted in the tests)."""
    return jnp.zeros((int(horizon), len(POLITICS_LEVERS)), dtype=jnp.float32)


def attach_politics_plan(state: GraphState, plan) -> GraphState:
    """Put the plan on the state as a dynamic pytree child. Values are data —
    two plans of the same horizon share one compiled program."""
    return state.update_global_attr(
        POLITICS_PLAN, jnp.asarray(plan, dtype=jnp.float32))


def external_intensity_of(levy_rate=0.0, culture_row=None,
                          cfg: PoliticsLeverConfig = PoliticsLeverConfig(),
                          levy_max: float = 0.3):
    """The siblings' bill, in the normalized intensity units column 4 expects.

    The levy term is ``levy_rate / levy_max`` — the spec's ``levy/0.3``. The
    culture term is ``culture_upkeep(row) / upkeep``, i.e. the culture family's
    own pricing function divided back out by the shared coefficient, so the two
    modules cannot drift apart on what a cultural lever costs. Pure arithmetic
    over plan rows: it touches no state, is called once per plan row when the
    server assembles the bundle, and never inside the traced step. The spec
    prices only the reach cap on the culture side; ``culture_upkeep`` prices all
    three culture levers. This follows the sibling module, which is the one that
    owns that call.
    """
    total = jnp.clip(jnp.asarray(levy_rate, dtype=jnp.float32),
                     0.0, levy_max) / levy_max
    if culture_row is not None:
        from .culture import culture_upkeep      # lazy: no import-order coupling
        total = total + culture_upkeep(culture_row) / cfg.upkeep
    return jnp.clip(total, 0.0, cfg.external_intensity_max)


def enforcement_rest(intensity, e_baseline: float = 0.5968,
                     repair_rate: float = 0.02,
                     cfg: PoliticsLeverConfig = PoliticsLeverConfig()):
    """First-order rest level of the enforcement stock under sustained upkeep —
    the receipt's "costs 0.10 enforcement at rest" line, and the shading rule for
    slider positions that drive the stock to zero.

    **Arithmetic, not measurement.** At rest ``update_regime``'s inflow balances
    its outflow: ``repair x (1 - e) + regime_rate x pressure - upkeep x I = 0``.
    Take the measured no-lever rest level ``e_baseline`` as pinning the lobby
    term, ``regime_rate x pressure = -repair x (1 - e_baseline)``, and the
    pressure term cancels::

        e* = e_baseline - (upkeep / repair_rate) x intensity

    which at the defaults is the spec's ``0.597 - 0.2 x intensity``. Lobby
    pressure is endogenous and will move with the plan, so this is a slider-time
    estimate that the engine's own run overrides. ``e_baseline`` is the
    three-families deposit's harbor/wait-path figure: 6 seeds, medians, **no
    CIs** — an ordering claim, not a magnitude that travels.
    """
    e = (jnp.asarray(e_baseline, dtype=jnp.float32)
         - (cfg.upkeep / max(float(repair_rate), 1e-9))
         * jnp.asarray(intensity, dtype=jnp.float32))
    return jnp.clip(e, 0.0, 1.0)


def make_politics_levers(cfg: PoliticsLeverConfig = PoliticsLeverConfig()):
    """The politics family's whole plan as one transform for the mechanism slot.

    Writes the enforcement stock, the two substrate rate globals
    ``update_regime`` reads next tick, the ``delegation`` ledger (sortition), and
    the wealth drip into the declared ``intervention_spend`` sink. Reads and
    writes nothing the culture family owns; shares only ``wealth`` with economy.

    Running in the mechanism slot means the two rate globals land **one tick
    later**, the same lag ``reach_cut_now`` already has. The receipt must not
    hide it.
    """

    @transform(reads=["step", POLITICS_PLAN, "wealth", "enforcement",
                      "delegation", "repair_rate_now", "entrenchment_gain_now"],
               writes=["wealth", "intervention_spend", "enforcement",
                       "delegation", "repair_rate_now", "entrenchment_gain_now"])
    def politics_levers(state: GraphState) -> GraphState:
        plan = state.global_attrs[POLITICS_PLAN]
        row = plan[jnp.clip(state.global_attrs["step"], 0, plan.shape[0] - 1)]
        # clipped in-transform as a second line of defence behind the server's
        # range whitelist; every clip is inactive at a neutral row
        spend_r = jnp.clip(row[0], 0.0, cfg.repair_spend_max)
        sort_r = jnp.clip(row[1], 0.0, cfg.sortition_max)
        rate_a = jnp.clip(row[2], 0.0, cfg.repair_rate_max)
        entr_a = jnp.clip(row[3], 0.0, cfg.entrenchment_gain_max)
        ext_i = jnp.clip(row[4], 0.0, cfg.external_intensity_max)

        is_human = state.node_types == 0
        h = is_human.astype(jnp.float32)
        n_h = jnp.maximum(jnp.sum(h), 1.0)

        # --- fund the office: household drip -> declared sink -> capacity ------
        # Money leaves the loop (it is one of ledger_society's five declared
        # wealth sinks) and buys a bounded uplift, 0.5 x drip_pc x (1 - e) --
        # the same purchase make_policy_levers and the fund-repair card make.
        wealth = state.node_attrs["wealth"]
        drip = spend_r * wealth * h
        wealth = wealth - drip
        enf = state.global_attrs["enforcement"]
        enf = enf + cfg.repair_efficiency * (jnp.sum(drip) / n_h) * (1.0 - enf)

        # --- the bill: this family's own act plus the siblings' -----------------
        # sortition is a political act and pays like one; the levy and the reach
        # cap arrive already normalized in column 4. Exactly 0 at a neutral row.
        intensity = sort_r / cfg.sortition_max + ext_i
        enf = jnp.clip(enf - cfg.upkeep * intensity, 0.0, 1.0)

        # --- seats by lot: blend citizen ballot rows toward the demos ----------
        # Row mass is conserved (the lottery rows are normalized and the
        # perturbation sums to zero across a row), the franchise floor on the
        # diagonal is untouched (both terms of ``delta`` are zero there), AI rows
        # are frozen, and the lottery runs over *citizens* only -- which is what
        # makes this the better version of churn rather than a duplicate of it:
        # churn re-draws over everyone, newcomers included.
        #
        # Written as ``D + sort_r * delta`` rather than ``make_policy_levers``'
        # algebraically equal ``diag + (1-s)*off + s*S*lottery``. The two agree
        # exactly at every slider position in exact arithmetic, but only this
        # form is *structurally* neutral at s = 0: the whole perturbation is
        # multiplied by the slider, so a zero row is ``D + 0.0``, whatever
        # rounding happened inside ``delta``. The decompose-and-recompose form
        # is exact eagerly and loses ~1 ULP on ``delegation`` under ``lax.scan``,
        # because XLA re-fuses the producer of ``D`` and does not evaluate the
        # two references to it identically -- measured 2026-07-31 with a literal
        # no-op ``diag + (D - diag)`` in the mechanism slot, which reproduces the
        # same 1.49e-08 drift. The pure tier is the tier that runs, so the
        # sealing convention has to hold there. ``make_policy_levers`` still
        # carries the old form and the weaker guarantee its test names
        # ("neutral to the ulp").
        D = state.adj_matrices["delegation"]
        N = D.shape[0]
        eye = jnp.eye(N)
        lottery = h[None, :] * (1.0 - eye)
        lottery = lottery / jnp.maximum(
            jnp.sum(lottery, axis=1, keepdims=True), 1e-12)
        off = D - jnp.diag(jnp.diag(D))
        delta = jnp.sum(off, axis=1, keepdims=True) * lottery - off
        D_out = jnp.where(is_human[:, None], D + sort_r * delta, D)

        # --- the two scenario dials: absolute values, 0 = hold ------------------
        repair_now = jnp.where(rate_a > 0.0, rate_a,
                               state.global_attrs["repair_rate_now"])
        entr_now = jnp.where(entr_a > 0.0, entr_a,
                             state.global_attrs["entrenchment_gain_now"])

        state = state.update_node_attrs("wealth", wealth)
        state = state.update_node_attrs("intervention_spend", drip)
        state = state.update_adj_matrix("delegation", D_out)
        state = state.update_global_attr("enforcement", enf)
        state = state.update_global_attr("repair_rate_now", repair_now)
        return state.update_global_attr("entrenchment_gain_now", entr_now)
    return politics_levers
