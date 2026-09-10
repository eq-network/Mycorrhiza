"""Behavioral tests for the politics family — SPEND.

Unit rungs on a mini state (direction and conservation), plus the two exactness
contracts this family owes: a neutral plan is a bit-exact identity — on the mini
state and on a real ``ledger_society`` run — and the ballot ledger's row sums are
conserved by sortition. Numbers are asserted only where the number *is* the
contract (the upkeep price line, the clip bounds, the rest-level arithmetic);
everything else is an ordering claim.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from cilib.core.graph import GraphState
from cilib.environments.ledger import row_stochastic_error
from cilib.environments.ledger_society import LedgerSocietyConfig, make_state
from cilib.environments.ledger_society.dynamics import build_step_fn
from cilib.mechanisms.families.culture import CULTURE_LEVERS, make_culture_levers
from cilib.mechanisms.families.economy import make_economy_levers
from cilib.mechanisms.families.politics import (
    POLITICS_LEVERS, POLITICS_PLAN, PoliticsLeverConfig, attach_politics_plan,
    enforcement_rest, external_intensity_of, make_politics_levers,
    neutral_politics_plan,
)

KEY = jr.PRNGKey(0)
CFG = PoliticsLeverConfig()

# four citizens + one AI actor; every row a ballot share distribution
D0 = jnp.array([
    [0.3, 0.5, 0.0, 0.0, 0.2],
    [0.1, 0.3, 0.3, 0.2, 0.1],
    [0.1, 0.1, 0.3, 0.4, 0.1],
    [0.2, 0.2, 0.1, 0.3, 0.2],
    [0.0, 0.0, 0.0, 0.0, 1.0],
], dtype=jnp.float32)
TYPES = jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32)


def _plan(repair_spend=0.0, sortition=0.0, repair_rate=0.0,
          entrenchment=0.0, external=0.0, horizon=4):
    row = jnp.array([repair_spend, sortition, repair_rate, entrenchment,
                     external], dtype=jnp.float32)
    return jnp.tile(row[None, :], (horizon, 1))


def _mini(plan, step=1, enforcement=0.5, wealth=(2.0, 2.0, 2.0, 2.0, 10.0),
          repair_rate_now=0.02, entrenchment_gain_now=0.0):
    n = len(wealth)
    state = GraphState(
        node_types=TYPES[:n],
        node_attrs={
            "wealth": jnp.asarray(wealth, dtype=jnp.float32),
            "intervention_spend": jnp.zeros(n, dtype=jnp.float32),
        },
        adj_matrices={"delegation": D0[:n, :n]},
        edge_attrs={},
        global_attrs={
            "step": jnp.array(step, dtype=jnp.int32),
            "enforcement": jnp.array(enforcement, dtype=jnp.float32),
            "repair_rate_now": jnp.array(repair_rate_now, dtype=jnp.float32),
            "entrenchment_gain_now": jnp.array(entrenchment_gain_now,
                                               dtype=jnp.float32),
        },
    )
    return attach_politics_plan(state, plan)


# --- the contract: a neutral plan changes nothing ---------------------------------

def test_neutral_plan_is_bit_exact_identity_on_every_field():
    """Including a non-default substrate rate: 0 in the two rate columns means
    *hold*, so the identity holds for any config, not just the defaults."""
    lever = make_politics_levers()
    state = _mini(neutral_politics_plan(4), step=3, enforcement=0.73,
                  repair_rate_now=0.037, entrenchment_gain_now=0.021)
    out = lever(state)
    for k in ("wealth", "intervention_spend"):
        assert bool(jnp.array_equal(out.node_attrs[k], state.node_attrs[k])), k
    assert bool(jnp.array_equal(out.adj_matrices["delegation"],
                                state.adj_matrices["delegation"]))
    for k in ("enforcement", "repair_rate_now", "entrenchment_gain_now"):
        assert bool(jnp.array_equal(out.global_attrs[k],
                                    state.global_attrs[k])), k


def test_neutral_plan_is_bit_exact_identity_on_a_real_run():
    """The rung that matters: 25 ticks of ledger_society with the transform in
    the mechanism slot must be indistinguishable from the model without it."""
    cfg = LedgerSocietyConfig()
    base_fn = build_step_fn(cfg, ())
    lever_fn = build_step_fn(cfg, (make_politics_levers(),))
    s_base = make_state(cfg, KEY)
    s_lever = attach_politics_plan(s_base, neutral_politics_plan(25))
    for k in jr.split(jr.PRNGKey(7), 25):
        s_base = base_fn(s_base, s_base.node_attrs["alloc_pref"], k)
        s_lever = lever_fn(s_lever, s_lever.node_attrs["alloc_pref"], k)
    for k, v in s_base.node_attrs.items():
        assert bool(jnp.array_equal(s_lever.node_attrs[k], v)), k
    for k, v in s_base.adj_matrices.items():
        assert bool(jnp.array_equal(s_lever.adj_matrices[k], v)), k
    for k, v in s_base.global_attrs.items():
        assert bool(jnp.array_equal(s_lever.global_attrs[k], v)), k


def test_neutral_plan_is_bit_exact_identity_under_lax_scan():
    """The eager loop above is the weaker rung: each tick is its own XLA program,
    so nothing fuses across the mechanism slot. Under ``lax.scan`` — the pure
    tier, the one that actually runs and the only one that vmaps over seeds — a
    whole tick is one fused program, and an identity written as
    ``diag + (D - diag)`` is NOT bit-exact there: XLA re-fuses the producer of
    ``D`` and the two references to it do not round alike (measured 2026-07-31,
    ~1 ULP on ``delegation`` at tick 1, still ~1 ULP at tick 60). This rung is
    what pins the ``D + sort_r * delta`` form, whose neutrality is structural."""
    cfg = LedgerSocietyConfig()
    n = 40

    def run(mechs, plan):
        st = make_state(cfg, KEY)
        if plan:
            st = attach_politics_plan(st, neutral_politics_plan(n))
        step_fn = build_step_fn(cfg, mechs)

        def body(s, k):
            return step_fn(s, s.node_attrs["alloc_pref"], k), None
        final, _ = lax.scan(body, st, jr.split(jr.PRNGKey(7), n))
        return final

    s_base = run((), False)
    s_lever = run((make_politics_levers(),), True)
    for k, v in s_base.node_attrs.items():
        assert bool(jnp.array_equal(s_lever.node_attrs[k], v)), k
    for k, v in s_base.adj_matrices.items():
        assert bool(jnp.array_equal(s_lever.adj_matrices[k], v)), k
    for k, v in s_base.global_attrs.items():
        assert bool(jnp.array_equal(s_lever.global_attrs[k], v)), k


def test_the_three_families_compose_and_a_neutral_bundle_seals_under_scan():
    """All three in one pipeline: the bundle's neutral plans are jointly a
    bit-exact identity under ``lax.scan``, and the only new state is the three
    plan arrays."""
    from cilib.mechanisms import (
        neutral_economy_plan, attach_economy_plan,
        neutral_culture_plan, attach_culture_plan,
    )
    cfg = LedgerSocietyConfig()
    n = 40

    def run(attach):
        st = make_state(cfg, KEY)
        mechs = ()
        if attach:
            mechs = (make_economy_levers(), make_culture_levers(),
                     make_politics_levers())
            st = attach_economy_plan(st, neutral_economy_plan(n))
            st = attach_culture_plan(st, neutral_culture_plan(n))
            st = attach_politics_plan(st, neutral_politics_plan(n))
        step_fn = build_step_fn(cfg, mechs)

        def body(s, k):
            return step_fn(s, s.node_attrs["alloc_pref"], k), None
        final, _ = lax.scan(body, st, jr.split(jr.PRNGKey(7), n))
        return final

    plain, bundled = run(False), run(True)
    for k, v in plain.node_attrs.items():
        assert bool(jnp.array_equal(bundled.node_attrs[k], v)), k
    for k, v in plain.adj_matrices.items():
        assert bool(jnp.array_equal(bundled.adj_matrices[k], v)), k
    for k, v in plain.global_attrs.items():
        assert bool(jnp.array_equal(bundled.global_attrs[k], v)), k
    assert (set(bundled.global_attrs) - set(plain.global_attrs)
            == {"economy_plan", "culture_plan", "politics_plan"})


# --- lever 1: fund the office -----------------------------------------------------

def test_fund_the_office_drains_household_wealth_and_buys_enforcement():
    lever = make_politics_levers()
    state = _mini(_plan(repair_spend=0.05), enforcement=0.5)
    out = lever(state)
    w_in, w_out = state.node_attrs["wealth"], out.node_attrs["wealth"]
    # the claimed direction: households poorer, the stock higher
    assert bool(jnp.all(w_out[:4] < w_in[:4]))
    assert float(out.global_attrs["enforcement"]) > 0.5
    # the AI hoard is not the office's funding source
    assert float(w_out[4]) == float(w_in[4])
    # more spending buys more capacity (monotone in the slider)
    small = lever(_mini(_plan(repair_spend=0.01), enforcement=0.5))
    assert (float(small.global_attrs["enforcement"])
            < float(out.global_attrs["enforcement"]))


def test_fund_the_office_conserves_the_money_it_removes_into_the_sink():
    """ledger_society's per-tick money identity counts intervention_spend as a
    declared sink: what leaves wealth must appear there, exactly."""
    lever = make_politics_levers()
    state = _mini(_plan(repair_spend=0.03))
    out = lever(state)
    removed = jnp.sum(state.node_attrs["wealth"]) - jnp.sum(
        out.node_attrs["wealth"])
    booked = jnp.sum(out.node_attrs["intervention_spend"])
    assert abs(float(removed - booked)) < 1e-6
    assert float(booked) > 0.0
    assert float(out.node_attrs["intervention_spend"][4]) == 0.0


# --- lever 2: seats by lot --------------------------------------------------------

def test_sortition_spreads_ballots_to_citizens_and_conserves_row_mass():
    lever = make_politics_levers()
    state = _mini(_plan(sortition=0.2))
    out = lever(state)
    D_out = out.adj_matrices["delegation"]
    # ballot mass moves off the concentrated delegate and off the AI actor,
    # onto citizens who held none -- the lottery runs over citizens only, which
    # is what distinguishes it from churn (churn re-draws over everyone)
    assert float(D_out[0, 1]) < float(D0[0, 1])
    assert float(D_out[0, 4]) < float(D0[0, 4])
    assert float(D_out[0, 2]) > float(D0[0, 2]) == 0.0
    # the franchise floor on the diagonal is untouched
    assert bool(jnp.allclose(jnp.diag(D_out), jnp.diag(D0), atol=1e-6))
    # conservation: every row is still a ballot share distribution
    assert row_stochastic_error(D_out) < 1e-5
    # AI rows are frozen
    assert bool(jnp.array_equal(D_out[4], D0[4]))
    # a bigger draw moves more mass (monotone in the slider)
    half = lever(_mini(_plan(sortition=0.1)))
    assert float(half.adj_matrices["delegation"][0, 1]) > float(D_out[0, 1])


# --- the bill: this family's own act, and the siblings' ---------------------------

def test_sortition_upkeep_debits_the_stock_at_the_quoted_price():
    lever = make_politics_levers()
    out = lever(_mini(_plan(sortition=0.10), enforcement=0.5))
    # the receipt's price line: 0.004 x (r / 0.20) per tick
    assert abs(float(out.global_attrs["enforcement"])
               - (0.5 - CFG.upkeep * 0.5)) < 1e-6
    assert float(out.global_attrs["enforcement"]) < 0.5


def test_external_intensity_bills_the_other_families_here():
    """The economy levy and the culture reach cap are political acts charged to
    the same stock; politics is the only transform that debits it."""
    lever = make_politics_levers()
    alone = lever(_mini(_plan(sortition=0.20), enforcement=0.6))
    billed = lever(_mini(_plan(sortition=0.20, external=2.0), enforcement=0.6))
    assert (float(billed.global_attrs["enforcement"])
            < float(alone.global_attrs["enforcement"]))
    assert abs(float(alone.global_attrs["enforcement"]
                     - billed.global_attrs["enforcement"])
               - CFG.upkeep * 2.0) < 1e-6
    # the stock is bounded below: sustained maximum intensity cannot go negative
    floored = lever(_mini(_plan(sortition=0.20, external=4.0),
                          enforcement=0.001))
    assert float(floored.global_attrs["enforcement"]) == 0.0


def test_external_intensity_of_reproduces_the_spec_formula():
    # levy alone: levy / 0.3
    assert abs(float(external_intensity_of(levy_rate=0.15)) - 0.5) < 1e-6
    assert float(external_intensity_of(levy_rate=0.0)) == 0.0
    # the culture term comes from the sibling's own pricing function, so the
    # full reach cap contributes exactly one normalized unit
    cut_row = jnp.zeros(len(CULTURE_LEVERS), dtype=jnp.float32).at[2].set(1.0)
    assert abs(float(external_intensity_of(levy_rate=0.15,
                                           culture_row=cut_row)) - 1.5) < 1e-5
    # a neutral culture row is free
    neutral_row = jnp.zeros(len(CULTURE_LEVERS), dtype=jnp.float32)
    assert float(external_intensity_of(culture_row=neutral_row)) == 0.0


# --- the two scenario dials -------------------------------------------------------

def test_repair_rate_column_raises_the_maintenance_rate_and_holds_at_zero():
    lever = make_politics_levers()
    raised = lever(_mini(_plan(repair_rate=0.04), repair_rate_now=0.02))
    assert float(raised.global_attrs["repair_rate_now"]) > 0.02
    held_in = _mini(_plan(), repair_rate_now=0.02)
    held = lever(held_in)
    assert bool(jnp.array_equal(held.global_attrs["repair_rate_now"],
                                held_in.global_attrs["repair_rate_now"]))


def test_entrenchment_gain_column_arms_the_hazard_and_holds_at_zero():
    lever = make_politics_levers()
    armed = lever(_mini(_plan(entrenchment=0.02), entrenchment_gain_now=0.0))
    assert float(armed.global_attrs["entrenchment_gain_now"]) > 0.0
    held_in = _mini(_plan(), entrenchment_gain_now=0.017)
    held = lever(held_in)
    assert bool(jnp.array_equal(held.global_attrs["entrenchment_gain_now"],
                                held_in.global_attrs["entrenchment_gain_now"]))


def test_the_scenario_dials_reach_the_substrate_next_tick():
    """The globals are only worth writing if update_regime reads them: an armed
    entrenchment gain must erode the stock on a concentrating ballot ledger, and
    a higher maintenance rate must push back against it."""
    cfg = LedgerSocietyConfig(entrenchment_threshold=0.01)   # concentration is
    lever_fn = build_step_fn(cfg, (make_politics_levers(),))  # certain to bind

    def run(entrenchment=0.0, repair_rate=0.0, n=40):
        s = attach_politics_plan(make_state(cfg, KEY),
                                 _plan(entrenchment=entrenchment,
                                       repair_rate=repair_rate, horizon=n))
        for k in jr.split(jr.PRNGKey(3), n):
            s = lever_fn(s, s.node_attrs["alloc_pref"], k)
        return float(s.global_attrs["enforcement"])

    armed = run(entrenchment=0.05)
    assert armed < run()                                    # doing nothing loses
    assert run(entrenchment=0.05, repair_rate=0.05) > armed  # repair pushes back


# --- ranges, tracing, and the disjoint-writes invariant ---------------------------

def test_out_of_range_plan_values_are_clipped_in_transform():
    lever = make_politics_levers()
    wild = lever(_mini(_plan(repair_spend=9.0, sortition=9.0, repair_rate=9.0,
                             entrenchment=9.0, external=9.0), enforcement=0.9))
    capped = lever(_mini(_plan(repair_spend=CFG.repair_spend_max,
                               sortition=CFG.sortition_max,
                               repair_rate=CFG.repair_rate_max,
                               entrenchment=CFG.entrenchment_gain_max,
                               external=CFG.external_intensity_max),
                         enforcement=0.9))
    assert bool(jnp.allclose(wild.node_attrs["wealth"],
                             capped.node_attrs["wealth"]))
    assert bool(jnp.allclose(wild.adj_matrices["delegation"],
                             capped.adj_matrices["delegation"]))
    for k in ("enforcement", "repair_rate_now", "entrenchment_gain_now"):
        assert abs(float(wild.global_attrs[k] - capped.global_attrs[k])) < 1e-7
    # negative values cannot pay the player back
    neg = lever(_mini(_plan(repair_spend=-1.0, sortition=-1.0, external=-1.0),
                      enforcement=0.5))
    assert float(neg.global_attrs["enforcement"]) == 0.5


def test_traces_under_jit_and_vmaps_over_seeds():
    cfg = LedgerSocietyConfig()
    step_fn = build_step_fn(cfg, (make_politics_levers(),))
    plan = _plan(repair_spend=0.02, sortition=0.1, external=0.5, horizon=12)

    def run(key):
        k0, kr = jr.split(key)
        s = attach_politics_plan(make_state(cfg, k0), plan)
        def body(state, k):
            state = step_fn(state, state.node_attrs["alloc_pref"], k)
            return state, state.global_attrs["enforcement"]
        final, series = lax.scan(body, s, jr.split(kr, 12))
        D = final.adj_matrices["delegation"]     # row_stochastic_error returns a
        err = jnp.max(jnp.abs(jnp.sum(D, axis=1) - 1.0))   # float; inline here
        return series[-1], err

    enf, err = jax.jit(jax.vmap(run))(jr.split(KEY, 3))
    assert enf.shape == (3,)
    assert bool(jnp.all((enf >= 0.0) & (enf <= 1.0)))
    assert float(jnp.max(err)) < 1e-4          # conservation survives the run


def test_family_writes_are_disjoint_except_the_shared_hoard():
    politics = make_politics_levers()
    culture = make_culture_levers()
    economy = make_economy_levers()
    assert not (set(politics.writes) & set(culture.writes))
    # the office is funded out of the same household hoard that funds
    # investment -- a physical overlap the spec's third failure mode depends on
    assert set(politics.writes) & set(economy.writes) == {"wealth"}
    assert POLITICS_PLAN not in set(culture.reads) | set(economy.reads)
    assert len(POLITICS_LEVERS) == neutral_politics_plan(2).shape[1]


def test_enforcement_rest_matches_the_spec_price_line():
    """e* = e_baseline - (upkeep / repair_rate) x intensity; at the defaults the
    coefficient is 0.2, the spec's arithmetic. An estimate, not a measurement."""
    assert abs(float(enforcement_rest(0.0)) - 0.5968) < 1e-5
    assert abs(float(enforcement_rest(1.0)) - (0.5968 - 0.2)) < 1e-5
    assert float(enforcement_rest(3.0)) == 0.0     # three levers at maximum
    assert float(enforcement_rest(2.0)) < float(enforcement_rest(1.0))
