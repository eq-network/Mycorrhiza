"""Behavioral tests for the economy (ALLOCATE) lever family.

Two rungs. Unit rungs run the transform on a mini state — direction, clipping,
conservation of the allocation row, the AI-rows-untouched contract. Integration
rungs run it in ``ledger_society``'s mechanism slot under ``lax.scan``, which is
where the one-tick lag and the compounding of investment are visible at all.

The exactness contract: a zero plan row is a bit-exact identity, both on the
mini state and across a whole run against the same run with no economy
transform and no plan on the state.

Imports reach into ``families.economy`` directly rather than through
``families/__init__.py`` so this file does not depend on the sibling families
being present.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.environments.ledger_society import LedgerSocietyConfig
from cilib.environments.ledger_society.dynamics import build_step_fn
from cilib.environments.ledger_society.state import make_state
from cilib.mechanisms.families.economy import (
    ECONOMY_LEVERS, ECONOMY_PLAN, EconomyLeverConfig, make_economy_levers,
    neutral_economy_plan, attach_economy_plan,
)

HUMAN_ALLOC = (0.70, 0.05, 0.03, 0.02, 0.20)
AI_ALLOC = (0.00, 0.50, 0.30, 0.10, 0.10)
SPENDS = ("consume_spend", "invest_spend", "broadcast_spend", "lobby_spend")


# --- unit rung ------------------------------------------------------------------

def _mini_state(row, step=7, wealth=(1.3, 0.7, 5.1), node_types=(0, 0, 1)):
    n = len(node_types)
    types = jnp.asarray(node_types, dtype=jnp.int32)
    pref = jnp.where(
        (types == 1)[:, None],
        jnp.array([AI_ALLOC], dtype=jnp.float32),
        jnp.array([HUMAN_ALLOC], dtype=jnp.float32))
    plan = jnp.tile(jnp.asarray([row], dtype=jnp.float32), (12, 1))
    return GraphState(
        node_types=types,
        node_attrs={
            "alloc_pref": pref,
            "wealth": jnp.asarray(wealth, dtype=jnp.float32),
        },
        adj_matrices={}, edge_attrs={},
        global_attrs={
            "step": jnp.array(step, dtype=jnp.int32),
            ECONOMY_PLAN: plan,
        },
    )


def test_plan_shape_and_neutral_builder():
    assert ECONOMY_LEVERS == ("d_consume", "d_invest", "d_broadcast",
                              "d_lobby", "levy_rate")
    plan = neutral_economy_plan(30)
    assert plan.shape == (30, 5)
    assert float(jnp.max(jnp.abs(plan))) == 0.0


def test_declared_effects_stay_out_of_the_other_families():
    lev = make_economy_levers()
    assert set(lev.writes) == {"alloc_pref", "wealth"}
    # the culture family owns the attention-kernel shape, the politics family
    # the enforcement stock and its dials — this family touches neither
    others = {"gamma_w_now", "update_rate_w_now", "reach_cut_now", "churn_now",
              "repair_rate_now", "entrenchment_gain_now", "enforcement",
              "listening", "delegation", "intervention_spend"}
    assert set(lev.writes) & others == set()
    assert ECONOMY_PLAN in lev.reads


def test_neutral_plan_is_bit_exact_identity():
    lev = make_economy_levers()
    state = _mini_state([0.0, 0.0, 0.0, 0.0, 0.0], step=123)
    out = lev(state)
    for k in ("alloc_pref", "wealth"):
        assert bool(jnp.array_equal(out.node_attrs[k], state.node_attrs[k])), k


def test_each_band_delta_moves_its_own_band_in_the_claimed_direction():
    lev = make_economy_levers()
    base = jnp.array(HUMAN_ALLOC)
    for col, (delta, sign) in enumerate([(-0.20, -1), (+0.20, +1),
                                         (+0.10, +1), (+0.10, +1)]):
        row = [0.0] * 5
        row[col] = delta
        out = lev(_mini_state(row))
        human = out.node_attrs["alloc_pref"][0]
        moved = float(human[col] - base[col])
        assert sign * moved > 0, (col, moved)
        assert abs(moved - delta) < 1e-6, (col, moved)
        # only that band and the residual move
        untouched = [c for c in range(4) if c != col]
        assert jnp.allclose(human[jnp.array(untouched)], base[jnp.array(untouched)])
        # the residual absorbs exactly what the band took
        assert abs(float(human[4] - base[4]) + delta) < 1e-6


def test_allocation_row_sums_to_one_and_ai_rows_are_never_touched():
    lev = make_economy_levers()
    for row in ([0.0, 0.15, 0.0, 0.0, 0.0],
                [-0.4, 0.0, 0.27, 0.18, 0.0],       # residual driven to zero
                [9.0, 9.0, 9.0, 9.0, 0.0]):         # wildly infeasible
        out = lev(_mini_state(row))
        pref = out.node_attrs["alloc_pref"]
        assert abs(float(jnp.sum(pref[0])) - 1.0) < 1e-5, row
        assert float(jnp.min(pref)) >= 0.0, row
        assert jnp.allclose(pref[2], jnp.array(AI_ALLOC)), row   # the AI row


def test_bands_clip_to_the_declared_ranges():
    cfg = EconomyLeverConfig()
    lev = make_economy_levers(cfg)
    high = lev(_mini_state([9.0, 9.0, 9.0, 9.0, 0.0])).node_attrs["alloc_pref"][0]
    low = lev(_mini_state([-9.0, -9.0, -9.0, -9.0, 0.0])).node_attrs["alloc_pref"][0]
    # every band inside [lo, hi]; the top row is additionally scaled back to a
    # feasible budget, so bands land at or below their ceiling
    for c in range(4):
        assert float(high[c]) <= cfg.band_hi[c] + 1e-6
        assert float(low[c]) >= cfg.band_lo[c] - 1e-6
        assert float(low[c]) <= cfg.band_lo[c] + 1e-6      # floor is binding
    assert float(jnp.sum(high[:4])) <= 1.0 + 1e-5


def test_levy_moves_wealth_from_ai_to_humans_and_conserves_the_total():
    lev = make_economy_levers()
    state = _mini_state([0.0, 0.0, 0.0, 0.0, 0.5],
                        wealth=(1.0, 1.0, 8.0), node_types=(0, 0, 1))
    out = lev(state).node_attrs["wealth"]
    assert float(out[2]) < 8.0                          # the AI stock falls
    assert float(out[0]) > 1.0 and float(out[1]) > 1.0  # humans gain
    assert abs(float(jnp.sum(out)) - 10.0) < 1e-5       # conserving transfer
    # clipped at levy_max: a plan asking for 5.0 does not confiscate 5x
    hot = make_economy_levers()(_mini_state([0.0] * 4 + [5.0],
                                            wealth=(1.0, 1.0, 8.0)))
    assert float(hot.node_attrs["wealth"][2]) >= 8.0 * (1.0 - EconomyLeverConfig().levy_max) - 1e-5
    assert abs(float(jnp.sum(hot.node_attrs["wealth"])) - 10.0) < 1e-5


# --- integration rung: the mechanism slot of ledger_society ----------------------

def _run(row=None, n_steps=40, seed=0, attach=True):
    """Pass-through closure (``SpendSharePolicy`` with noise 0) under lax.scan.
    Returns the final state and per-tick sums over the human rows."""
    cfg = LedgerSocietyConfig()
    H = cfg.n_humans
    mechs = (make_economy_levers(),) if attach else ()
    step_fn = build_step_fn(cfg, mechs)
    state = make_state(cfg, jr.PRNGKey(seed))
    if attach:
        plan = (neutral_economy_plan(n_steps) if row is None
                else jnp.tile(jnp.asarray([row], dtype=jnp.float32), (n_steps, 1)))
        state = attach_economy_plan(state, plan)

    def body(st, k):
        nxt = step_fn(st, st.node_attrs["alloc_pref"], k)   # pass-through policy
        series = {n: jnp.sum(nxt.node_attrs[n][:H]) for n in SPENDS}
        series["capital"] = jnp.sum(nxt.node_attrs["capital"][:H])
        series["wealth"] = jnp.sum(nxt.node_attrs["wealth"][:H])
        return nxt, series

    return jax.lax.scan(body, state, jr.split(jr.PRNGKey(seed + 1), n_steps))


def test_neutral_plan_run_is_bit_identical_to_the_plan_being_absent():
    with_plan, _ = _run(row=None, n_steps=40, attach=True)
    without, _ = _run(n_steps=40, attach=False)
    for k in ("wealth", "capital", "alloc_pref", "consume_spend", "invest_spend",
              "broadcast_spend", "lobby_spend", "belief", "influence"):
        assert bool(jnp.array_equal(with_plan.node_attrs[k],
                                    without.node_attrs[k])), k
    for k in ("listening", "delegation"):
        assert bool(jnp.array_equal(with_plan.adj_matrices[k],
                                    without.adj_matrices[k])), k
    for k in ("enforcement", "efficiency", "policy_target"):
        assert bool(jnp.array_equal(with_plan.global_attrs[k],
                                    without.global_attrs[k])), k


def test_a_change_lands_the_following_tick_not_this_one():
    """The levers run after ``allocate``, so tick 0 is untouched and tick 1
    carries the move — the lag the spec says must not be hidden."""
    _, base = _run(row=None, n_steps=6)
    _, hot = _run(row=[0.0, 0.20, 0.0, 0.0, 0.0], n_steps=6)
    assert float(base["invest_spend"][0]) == float(hot["invest_spend"][0])
    assert float(hot["invest_spend"][1]) > float(base["invest_spend"][1])


def test_invest_delta_raises_investment_and_compounds_into_human_capital():
    final_b, base = _run(row=None, n_steps=40)
    final_h, hot = _run(row=[0.0, 0.20, 0.0, 0.0, 0.0], n_steps=40)
    assert float(hot["invest_spend"][1]) > float(base["invest_spend"][1])
    assert float(hot["capital"][-1]) > float(base["capital"][-1])
    H = LedgerSocietyConfig().n_humans
    assert (float(jnp.sum(final_h.node_attrs["capital"][:H]))
            > float(jnp.sum(final_b.node_attrs["capital"][:H])))


def test_consume_delta_lowers_the_living_band_in_money():
    _, base = _run(row=None, n_steps=40)
    _, cold = _run(row=[-0.20, 0.0, 0.0, 0.0, 0.0], n_steps=40)
    assert float(cold["consume_spend"][1]) < float(base["consume_spend"][1])
    # what is not consumed is not destroyed: the residual is the hoard
    assert float(cold["wealth"][-1]) > float(base["wealth"][-1])


def test_broadcast_delta_raises_bought_reach():
    _, base = _run(row=None, n_steps=40)
    _, hot = _run(row=[0.0, 0.0, 0.15, 0.0, 0.0], n_steps=40)
    assert float(hot["broadcast_spend"][1]) > float(base["broadcast_spend"][1])
    assert float(jnp.mean(hot["broadcast_spend"])) > float(
        jnp.mean(base["broadcast_spend"]))


def test_lobby_delta_raises_funded_pressure():
    _, base = _run(row=None, n_steps=40)
    _, hot = _run(row=[0.0, 0.0, 0.0, 0.15, 0.0], n_steps=40)
    assert float(hot["lobby_spend"][1]) > float(base["lobby_spend"][1])
    assert float(jnp.mean(hot["lobby_spend"])) > float(jnp.mean(base["lobby_spend"]))


def test_levy_conserves_total_wealth_across_a_whole_run():
    plain, _ = _run(row=None, n_steps=40)
    levied, _ = _run(row=[0.0, 0.0, 0.0, 0.0, 0.25], n_steps=40)
    H = LedgerSocietyConfig().n_humans
    assert (float(jnp.sum(levied.node_attrs["wealth"][H:]))
            < float(jnp.sum(plain.node_attrs["wealth"][H:])))     # AI stock down
    assert (float(jnp.sum(levied.node_attrs["wealth"][:H]))
            > float(jnp.sum(plain.node_attrs["wealth"][:H])))     # humans up


def test_plan_values_vmap_over_seeds_and_over_plans():
    """Values are data: one compiled program, many seeds and many plans."""
    cfg = LedgerSocietyConfig()
    step_fn = build_step_fn(cfg, (make_economy_levers(),))
    n_steps = 8

    def run(seed, row):
        state = attach_economy_plan(
            make_state(cfg, jr.PRNGKey(0)),
            jnp.tile(row[None, :], (n_steps, 1)))
        state = state.update_global_attr("rng_key", jr.PRNGKey(seed))

        def body(st, k):
            return step_fn(st, st.node_attrs["alloc_pref"], k), None
        final, _ = jax.lax.scan(body, state, jr.split(jr.PRNGKey(seed), n_steps))
        return jnp.sum(final.node_attrs["capital"])

    rows = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.2, 0.0, 0.0, 0.0],
                      [-0.1, 0.1, 0.05, 0.05, 0.1]], dtype=jnp.float32)
    out = jax.jit(jax.vmap(jax.vmap(run, in_axes=(None, 0)), in_axes=(0, None)))(
        jnp.arange(3), rows)
    assert out.shape == (3, 3)
    assert bool(jnp.all(out[:, 1] > out[:, 0]))     # invest builds capital
