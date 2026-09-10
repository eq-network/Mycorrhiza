"""Behavioral tests for the culture family — SHAPE.

The levers write substrate globals, so every behavioral rung runs the real
attention path from ``ledger_society.dynamics`` on top of them: the lever writes
``gamma_w_now`` / ``update_rate_w_now`` / ``reach_cut_now``, the unmodified
``broadcast_reach -> rewire_listening -> listen_influence_update`` reads them,
and the assertion is about the ledger that moved. Directions and conservation
only, except the neutrality contract, which is bit-exact by design.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments.ledger_society.config import LedgerSocietyConfig
from cilib.environments.ledger_society.state import make_state
from cilib.environments.ledger_society.dynamics import (
    make_broadcast_reach, make_listen_influence_update, make_rewire_listening,
)
from cilib.metrics.families.concentration import hhi_of
from cilib.mechanisms.families.culture import (
    CULTURE_LEVERS, CultureLeverConfig, attach_culture_plan, culture_upkeep,
    make_culture_levers, neutral_culture_plan,
)

CFG = LedgerSocietyConfig()
LEVER_CFG = CultureLeverConfig.from_society(CFG)


def _society(plan_row=None, horizon=8, spend=0.0):
    """A real ledger_society state with a spread attention distribution.

    Attention influence starts uniform, and a uniform distribution is a fixed
    point of the exponent — every node is equally attractive at any gamma — so
    the shape levers are invisible until the town has some spread. Spreading it
    is what the arriving actors do; here it is set directly.
    """
    state = make_state(CFG, jr.PRNGKey(0))
    N = state.node_types.shape[0]
    v = jnp.arange(1, N + 1, dtype=jnp.float32)
    state = state.update_node_attrs("listen_influence", v / jnp.sum(v))
    state = state.update_node_attrs(
        "broadcast_spend", jnp.full((N,), spend, dtype=jnp.float32))
    if plan_row is not None:
        plan = jnp.tile(jnp.asarray(plan_row, dtype=jnp.float32)[None, :],
                        (horizon, 1))
        state = attach_culture_plan(state, plan)
    return state


def _attention_path(state, lever, n_ticks):
    """n ticks of lever -> reach -> rewire -> influence, stepping the clock."""
    reach = make_broadcast_reach(CFG)
    rewire = make_rewire_listening(CFG)
    influence = make_listen_influence_update(CFG)
    for _ in range(n_ticks):
        if lever is not None:
            state = lever(state)
        state = influence(rewire(reach(state)))
        state = state.update_global_attr("step", state.global_attrs["step"] + 1)
    return state


# --- the neutrality contract (bit-exact, not a direction) ------------------------

def test_neutral_plan_is_bit_exact_identity():
    lever = make_culture_levers(LEVER_CFG)
    neutral = attach_culture_plan(_society(plan_row=None),
                                  neutral_culture_plan(8))
    absent = _society(plan_row=None)

    once = lever(neutral)
    for k in ("gamma_w_now", "update_rate_w_now", "reach_cut_now"):
        assert bool(jnp.array_equal(once.global_attrs[k],
                                    absent.global_attrs[k])), k

    # and the whole downstream ledger, 12 ticks in: a neutral plan is the
    # pre-family model bit for bit, pow path included
    with_lever = _attention_path(neutral, lever, 12)
    without = _attention_path(absent, None, 12)
    assert bool(jnp.array_equal(with_lever.adj_matrices["listening"],
                                without.adj_matrices["listening"]))
    assert bool(jnp.array_equal(with_lever.node_attrs["listen_influence"],
                                without.node_attrs["listen_influence"]))


# --- one behavioral assertion per lever ------------------------------------------

def test_gamma_delta_shapes_the_square():
    """gamma up concentrates attention, gamma down spreads it — the claim the
    lever is named for. Instrument: HHI of listen_influence, 1/N at an equal
    split, 1.0 at a monopoly."""
    lever = make_culture_levers(LEVER_CFG)
    sharp = _attention_path(_society([+0.4, 0.0, 0.0]), lever, 25)
    flat = _attention_path(_society([-0.4, 0.0, 0.0]), lever, 25)
    base = _attention_path(_society([0.0, 0.0, 0.0]), lever, 25)

    h_sharp = float(hhi_of(sharp.node_attrs["listen_influence"]))
    h_flat = float(hhi_of(flat.node_attrs["listen_influence"]))
    h_base = float(hhi_of(base.node_attrs["listen_influence"]))
    assert h_sharp > h_base > h_flat


def test_update_rate_delta_sets_how_fast_the_town_turns():
    """The turn-rate lever moves the attention ledger further per tick in the
    same direction, and does not change where it is heading: the time constant
    is 1/rate, so a faster town covers more of the same distance."""
    lever = make_culture_levers(LEVER_CFG)
    before = _society([0.0, 0.0, 0.0]).adj_matrices["listening"]

    def moved(delta):
        out = _attention_path(_society([0.0, delta, 0.0]), lever, 1)
        return out.adj_matrices["listening"] - before

    fast, slow = moved(+0.07), moved(-0.05)
    d_fast = float(jnp.sum(jnp.abs(fast)))
    d_slow = float(jnp.sum(jnp.abs(slow)))
    assert d_fast > d_slow > 0.0
    # same heading, longer stride: the two steps are collinear
    cos = float(jnp.sum(fast * slow)
                / (jnp.linalg.norm(fast) * jnp.linalg.norm(slow)))
    assert cos > 0.999


def test_reach_cut_scales_the_money_to_attention_channel():
    """The cap is the one culture lever that acts on the same tick: it scales
    bought reach, so a spender's attractiveness boost falls toward 1.0."""
    lever = make_culture_levers(LEVER_CFG)
    reach = make_broadcast_reach(CFG)
    boost = {}
    for cut in (0.0, 0.5, 1.0):
        out = reach(lever(_society([0.0, 0.0, cut], spend=0.25)))
        boost[cut] = float(jnp.mean(out.node_attrs["attract_boost"]))
    assert boost[0.0] > boost[0.5] > boost[1.0]
    assert abs(boost[1.0] - 1.0) < 1e-6            # fully capped: reach unbought


# --- conservation: what this family may not break --------------------------------

def test_attention_rows_stay_conserved_at_every_plan_extreme():
    """The culture ledger's conservation law is per-row attention share. The
    shape levers change how the kernel reallocates, never how much there is —
    including outside the declared ranges, where the clip catches the row."""
    lever = make_culture_levers(LEVER_CFG)
    rows = [[0.0, 0.0, 0.0], [+0.4, +0.07, 1.0], [-0.4, -0.05, 0.0],
            [9.0, 5.0, 3.0], [-9.0, -5.0, -3.0]]
    for row in rows:
        out = _attention_path(_society(row, spend=0.3), lever, 10)
        sums = jnp.sum(out.adj_matrices["listening"], axis=1)
        assert float(jnp.max(jnp.abs(sums - 1.0))) < 1e-5, row
        v = out.node_attrs["listen_influence"]
        assert abs(float(jnp.sum(v)) - 1.0) < 1e-5, row
        assert float(jnp.min(v)) >= 0.0, row


def test_out_of_range_plan_is_clipped_to_the_declared_ranges():
    lever = make_culture_levers(LEVER_CFG)
    hi = lever(_society([9.0, 5.0, 3.0]))
    assert abs(float(hi.global_attrs["gamma_w_now"]) - 1.4) < 1e-6
    assert abs(float(hi.global_attrs["update_rate_w_now"]) - 0.15) < 1e-6
    assert abs(float(hi.global_attrs["reach_cut_now"]) - 1.0) < 1e-6
    lo = lever(_society([-9.0, -5.0, -3.0]))
    assert abs(float(lo.global_attrs["gamma_w_now"]) - 0.6) < 1e-6
    assert abs(float(lo.global_attrs["update_rate_w_now"]) - 0.03) < 1e-6
    assert float(lo.global_attrs["reach_cut_now"]) == 0.0


def test_plan_is_read_by_row_per_tick_and_clamps_past_the_horizon():
    lever = make_culture_levers(LEVER_CFG)
    plan = jnp.zeros((4, 3), dtype=jnp.float32).at[2, 2].set(0.75)
    state = _society(plan_row=[0.0, 0.0, 0.0]).update_global_attr(
        "culture_plan", plan)
    cuts = []
    for t in (0, 2, 9):
        at_t = state.update_global_attr("step", jnp.array(t, dtype=jnp.int32))
        cuts.append(float(lever(at_t).global_attrs["reach_cut_now"]))
    assert cuts[0] == 0.0                 # row 0: neutral
    assert abs(cuts[1] - 0.75) < 1e-6     # row 2: the one live row
    assert cuts[2] == 0.0                 # past the horizon: the last row holds


# --- the cost model this family exports but does not charge ----------------------

def test_upkeep_is_zero_at_neutral_and_symmetric_in_the_delta():
    assert float(culture_upkeep(jnp.zeros(3), LEVER_CFG)) == 0.0
    down = float(culture_upkeep(jnp.array([-0.4, 0.0, 0.0]), LEVER_CFG))
    up = float(culture_upkeep(jnp.array([+0.4, 0.0, 0.0]), LEVER_CFG))
    assert down > 0.0 and abs(down - up) < 1e-9
    both = float(culture_upkeep(jnp.array([+0.4, -0.05, 0.0]), LEVER_CFG))
    assert both > up


# --- the plan is data: one program, every plan ------------------------------------

def test_plan_values_vmap_without_recompiling():
    lever = make_culture_levers(LEVER_CFG)
    state = _society([0.0, 0.0, 0.0])
    plans = jnp.stack([
        jnp.tile(jnp.array([g, r, c], dtype=jnp.float32)[None, :], (8, 1))
        for g, r, c in ((0.0, 0.0, 0.0), (0.4, 0.07, 1.0), (-0.4, -0.05, 0.5))
    ])

    @jax.jit
    def run(plan):
        out = _attention_path(state.update_global_attr("culture_plan", plan),
                              lever, 5)
        return (jnp.array([out.global_attrs["gamma_w_now"],
                           out.global_attrs["update_rate_w_now"],
                           out.global_attrs["reach_cut_now"]]),
                out.adj_matrices["listening"])

    knobs, W = jax.vmap(run)(plans)
    assert knobs.shape == (3, 3) and len(CULTURE_LEVERS) == plans.shape[-1]
    assert jnp.allclose(knobs, jnp.array([[1.0, 0.08, 0.0],
                                          [1.4, 0.15, 1.0],
                                          [0.6, 0.03, 0.5]]), atol=1e-6)
    # one traced program, three different towns
    assert float(jnp.max(jnp.abs(W[1] - W[0]))) > 1e-4
    assert float(jnp.max(jnp.abs(W[2] - W[0]))) > 1e-4


def test_writes_stay_out_of_the_other_families_fields():
    lever = make_culture_levers(LEVER_CFG)
    assert set(lever.writes) == {"gamma_w_now", "update_rate_w_now",
                                 "reach_cut_now"}
    assert set(lever.reads) == {"step", "culture_plan"}
