"""
Validation ladder for the Ledger Society — conservation, sealing, agnosticism,
and the emergent flywheel. Behavioral rungs (direction/ordering), never
bit-exact numbers, except where the contract IS exactness (conservation,
sealing).
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

from cilib.environments import make_env
from cilib.environments.attachment import preferential_reallocation
from cilib.environments.ledger import row_stochastic_error
from cilib.environments.ledger_society import LedgerSocietyConfig, make_state

KEY = jr.PRNGKey(0)

# the three channel dials at zero — every cross-domain edge sealed
SEALED = dict(reach_per_spend=0.0, attention_to_ballots=0.0, regime_rate=0.0,
              entrenchment_gain=0.0)


def _run(n_steps=200, n_seeds=4, **cfg):
    env = make_env("ledger_society", **cfg)
    finals, traces = env.run_batch(KEY, n_seeds=n_seeds, n_steps=n_steps)
    return env, finals, traces


# --- rung 1: the kernel is pure reallocation --------------------------------------

def test_kernel_conserves_rows_and_respects_freeze():
    cfg = LedgerSocietyConfig()
    state = make_state(cfg, KEY)
    W = state.adj_matrices["listening"]
    a = jnp.linspace(1.0, 2.0, W.shape[0])
    frozen = state.node_types == 1
    W2 = preferential_reallocation(W, a, 0.1, 0.15, churn=0.05, frozen_rows=frozen)
    assert row_stochastic_error(W2) < 1e-5
    assert bool(jnp.all(jnp.where(frozen[:, None], W2 == W, True)))


# --- rung 2: ledger contracts hold on a real run -----------------------------------

def test_money_conservation_and_adjacency_ledgers():
    env, finals, traces = _run(n_steps=60, n_seeds=2)
    # per-tick identity: Δ(Σ wealth) == Σ income − Σ spends  (post-tick trace)
    w = jnp.sum(traces["wealth"], axis=-1)                    # (seeds, T)
    mint = jnp.sum(traces["last_income"], axis=-1)
    sunk = sum(jnp.sum(traces[k], axis=-1) for k in
               ("consume_spend", "invest_spend", "broadcast_spend", "lobby_spend",
                "intervention_spend"))
    resid = (w[:, 1:] - w[:, :-1]) - (mint[:, 1:] - sunk[:, 1:])
    scale = jnp.maximum(jnp.max(jnp.abs(w)), 1.0)
    assert float(jnp.max(jnp.abs(resid))) / float(scale) < 1e-4
    # adjacency ledgers stay row-stochastic to the end
    assert row_stochastic_error(finals.adj_matrices["listening"][0]) < 1e-4
    assert row_stochastic_error(finals.adj_matrices["delegation"][0]) < 1e-4
    # ballot power is a share distribution every tick
    p = jnp.sum(traces["influence"], axis=-1)
    assert float(jnp.max(jnp.abs(p - 1.0))) < 1e-4


def test_top_target_trace_indexes_dominant_edges():
    env, finals, traces = _run(n_steps=60, n_seeds=2)
    N = finals.adj_matrices["listening"].shape[-1]
    for field, ledger in (("top_listen_target", "listening"),
                          ("top_delegate_target", "delegation")):
        idx = traces[field]                                   # (seeds, T, N)
        assert idx.dtype == jnp.int32
        assert bool(jnp.all((idx >= 0) & (idx < N)))
        # the self column is masked out of the argmax
        assert bool(jnp.all(idx != jnp.arange(N)[None, None, :]))
        # trace is post-pipeline per tick: the last row must equal the masked
        # argmax of the final ledger — exactness, not ordering
        A = finals.adj_matrices[ledger]                       # (seeds, N, N)
        masked = jnp.where(jnp.eye(N, dtype=bool)[None], -jnp.inf, A)
        assert bool(jnp.all(idx[:, -1] == jnp.argmax(masked, axis=-1)))


# --- rung 3: sealing — an economy dial cannot move culture or politics ------------

def test_sealed_domains_are_bit_identical_under_economy_dial_change():
    _, _, tA = _run(n_steps=80, n_seeds=2, prosperity_gain=1.0, **SEALED)
    _, _, tB = _run(n_steps=80, n_seeds=2, prosperity_gain=3.0, **SEALED)
    for field in ("belief", "listen_influence", "influence"):
        assert bool(jnp.array_equal(tA[field], tB[field])), field
    # and the economy did actually change (the dial is live)
    assert not bool(jnp.array_equal(tA["last_income"], tB["last_income"]))


# --- rung 4: value-agnosticism — the reach channel answers to spending, not type --

def test_humans_who_buy_reach_gain_attention():
    # channel isolation: no AI arrivals, so humans are the only spenders — the
    # reach channel must answer to their spending exactly as it would to any
    # agent's (at defaults the same purchase is real but drowned by AI budgets
    # 10-40x larger, which is the disempowerment dynamic itself, not a bug)
    _, _, base = _run(first_arrival=9999)
    _, _, spend = _run(first_arrival=9999, human_alloc=(0.40, 0.05, 0.30, 0.05, 0.20))
    share = lambda tr: float(jnp.mean(
        jnp.sum(tr["listen_influence"][:, -50:, :20], axis=-1)))
    assert share(spend) > share(base) + 0.02


def test_ai_that_never_broadcasts_captures_less_attention():
    _, _, base = _run()
    _, _, silent = _run(ai_alloc=(0.0, 0.6, 0.0, 0.1, 0.3))
    human_share = lambda tr: float(jnp.mean(
        jnp.sum(tr["listen_influence"][:, -50:, :20], axis=-1)))
    assert human_share(silent) > human_share(base) + 0.02


# --- rung 4b: intervention events (docs/gd-game-design.md, 2026-07-31) ------------

def test_reach_cut_zero_is_bit_identical():
    # the influence-cap card at 0 is the pre-card model, exactly (sealing
    # convention: an unenacted card must not perturb a single bit)
    _, _, tA = _run(n_steps=80, n_seeds=2)
    _, _, tB = _run(n_steps=80, n_seeds=2, reach_cut=0.0, reach_cut_onset=40)
    for field in tA:
        assert bool(jnp.array_equal(tA[field], tB[field])), field


def test_reach_cut_slows_ai_attention_capture():
    _, _, base = _run(n_steps=200)
    _, _, cut = _run(n_steps=200, reach_cut=0.9, reach_cut_onset=30)
    human_share = lambda tr: float(jnp.mean(
        jnp.sum(tr["listen_influence"][:, -50:, :20], axis=-1)))
    assert human_share(cut) > human_share(base) + 0.02


def test_intervention_plan_conserves_and_repairs():
    from cilib.mechanisms import InterventionPlanConfig, make_interventions

    plan = make_interventions(InterventionPlanConfig(
        levy_onset=40, levy_rate=0.10,
        repair_onset=40, repair_spend_rate=0.02, repair_efficiency=0.5,
        enforcement_debits=((40, 0.15),)))
    env = make_env("ledger_society", mechanisms=(plan,), regime_rate=0.03)
    _, tr = env.run_batch(KEY, n_seeds=2, n_steps=200)
    envb = make_env("ledger_society", regime_rate=0.03)
    _, base = envb.run_batch(KEY, n_seeds=2, n_steps=200)

    # conservation holds with the levy transfer + drip sink in the loop
    w = jnp.sum(tr["wealth"], axis=-1)
    mint = jnp.sum(tr["last_income"], axis=-1)
    sunk = sum(jnp.sum(tr[k], axis=-1) for k in
               ("consume_spend", "invest_spend", "broadcast_spend", "lobby_spend",
                "intervention_spend"))
    resid = (w[:, 1:] - w[:, :-1]) - (mint[:, 1:] - sunk[:, 1:])
    assert float(jnp.max(jnp.abs(resid))) / float(
        jnp.maximum(jnp.max(jnp.abs(w)), 1.0)) < 1e-4
    # the levy moves late human wealth share up (direction, not magnitude)
    hshare = lambda t: float(jnp.mean(
        jnp.sum(t["wealth"][:, -50:, :20], axis=-1)
        / jnp.maximum(jnp.sum(t["wealth"][:, -50:, :], axis=-1), 1e-12)))
    assert hshare(tr) > hshare(base)
    # funded repair holds late enforcement above the undefended run
    assert (float(jnp.mean(tr["enforcement"][:, -50:]))
            > float(jnp.mean(base["enforcement"][:, -50:])))
    # the debit is visible at its tick: enforcement drops from t=40 to t=41
    assert float(jnp.mean(tr["enforcement"][:, 41] - tr["enforcement"][:, 39])) < 0.0


def test_empty_intervention_plan_is_bit_identical():
    from cilib.mechanisms import InterventionPlanConfig, make_interventions

    plan = make_interventions(InterventionPlanConfig())
    env = make_env("ledger_society", mechanisms=(plan,))
    _, tr = env.run_batch(KEY, n_seeds=2, n_steps=80)
    envb = make_env("ledger_society")
    _, base = envb.run_batch(KEY, n_seeds=2, n_steps=80)
    for field in base:
        assert bool(jnp.array_equal(tr[field], base[field])), field


# --- rung 4c: live policy levers (docs/remote-engine-design.md, 2026-07-31) -------

def _run_planned(plan, n_steps, **cfg):
    from cilib.core.scan import run_scan
    env = make_env("ledger_society", policy_horizon=n_steps, **cfg)
    k_init, k_run = jr.split(KEY)
    state = env.init_fn(k_init).update_global_attr("policy_plan", plan)
    return run_scan(env.round_fn, state, n_steps, k_run, trace_fn=env.trace_fn)


def test_zero_policy_plan_is_neutral_to_the_ulp():
    # the lever math at 0 is elementwise exact (x1.0, +0.0), but the plan is a
    # DYNAMIC input — XLA cannot constant-fold it away, compiles a different
    # program, and its fusion rounds the delegation->influence normalization
    # path by ~1 ULP. So the zero-plan contract is ULP-tight equivalence, not
    # bit identity (which the static dials do keep, rungs 3 and 4b).
    T = 80
    _, tr = _run_planned(jnp.zeros((T, 4), dtype=jnp.float32), n_steps=T)
    _, base = make_env("ledger_society").run(KEY, T)
    for field in base:
        a, b = tr[field], base[field]
        if jnp.issubdtype(a.dtype, jnp.integer):
            assert bool(jnp.array_equal(a, b)), field
        else:
            scale = jnp.maximum(jnp.max(jnp.abs(b)), 1.0)
            assert float(jnp.max(jnp.abs(a - b))) / float(scale) < 1e-6, field


def test_levy_lever_moves_wealth_share():
    T = 200
    plan = jnp.zeros((T, 4), dtype=jnp.float32).at[100:, 0].set(0.15)
    _, tr = _run_planned(plan, n_steps=T)
    _, base = _run_planned(jnp.zeros((T, 4), dtype=jnp.float32), n_steps=T)
    hws = lambda t: float(jnp.mean(
        jnp.sum(t["wealth"][-50:, :20], axis=-1)
        / jnp.maximum(jnp.sum(t["wealth"][-50:, :], axis=-1), 1e-12)))
    assert hws(tr) > hws(base)


def test_reach_cut_lever_defends_attention():
    T = 200
    plan = jnp.zeros((T, 4), dtype=jnp.float32).at[30:, 2].set(0.9)
    _, tr = _run_planned(plan, n_steps=T)
    _, base = _run_planned(jnp.zeros((T, 4), dtype=jnp.float32), n_steps=T)
    share = lambda t: float(jnp.mean(
        jnp.sum(t["listen_influence"][-50:, :20], axis=-1)))
    assert share(tr) > share(base) + 0.02


def test_sortition_lever_keeps_rows_and_lifts_power():
    T = 200
    plan = jnp.zeros((T, 4), dtype=jnp.float32).at[30:, 3].set(0.1)
    finals, tr = _run_planned(plan, n_steps=T)
    _, base = _run_planned(jnp.zeros((T, 4), dtype=jnp.float32), n_steps=T)
    assert row_stochastic_error(finals.adj_matrices["delegation"]) < 1e-4
    power = lambda t: float(jnp.mean(jnp.sum(t["influence"][-50:, :20], axis=-1)))
    assert power(tr) > power(base)


def test_policy_upkeep_drains_enforcement_early():
    # full-intensity reach cut from t=0: before the dynamics diverge, the only
    # difference is the upkeep drain — early enforcement sits strictly below
    T = 30
    plan = jnp.zeros((T, 4), dtype=jnp.float32).at[:, 2].set(1.0)
    _, tr = _run_planned(plan, n_steps=T)
    _, base = _run_planned(jnp.zeros((T, 4), dtype=jnp.float32), n_steps=T)
    assert (float(jnp.mean(tr["enforcement"][1:10]))
            < float(jnp.mean(base["enforcement"][1:10])))


# --- rung 4d: the reservoir's insularity is a dial, not a hidden constant ---------

def test_ai_insularity_zero_is_bit_identical():
    _, _, tA = _run(n_steps=80, n_seeds=2)
    _, _, tB = _run(n_steps=80, n_seeds=2, ai_insularity=0.0)
    for field in tA:
        assert bool(jnp.array_equal(tA[field], tB[field])), field


def test_insular_reservoir_removes_the_attention_floor():
    """Frozen AI rows sit at the t=0 draw, which points ~0.69 of an AI actor's
    attention at humans forever — a floor under the human attention share that
    NO other dial moves (probed 2026-08-01: 0.4505 at maximum channel dials,
    unchanged with the diagonal floors at zero). Redirecting those rows into the
    AI block removes it, which is what makes full capture expressible."""
    _, _, open_ = _run(n_steps=300, reach_per_spend=8.0)
    _, _, insular = _run(n_steps=300, reach_per_spend=8.0, ai_insularity=1.0)
    share = lambda tr: float(jnp.mean(
        jnp.sum(tr["listen_influence"][:, -75:, :20], axis=-1)))
    assert share(open_) > 0.3            # the floor is there without the dial
    assert share(insular) < 0.05         # and gone with it


def test_everything_off_lets_all_three_ledgers_be_captured():
    """The honest opposite of rung 6: with the reservoir insular, the channels
    at their maxima, no institutional upkeep and no diagonal floors, humans can
    lose all three ledgers. A model that cannot express this cannot be said to
    have found that they hold."""
    _, _, tr = _run(n_steps=400, ai_insularity=1.0, reach_per_spend=8.0,
                    attention_to_ballots=4.0, regime_rate=0.04,
                    repair_rate=0.0, self_weight_w=0.0, self_weight_d=0.0)
    inc = float(jnp.mean(jnp.sum(tr["last_income"][:, -100:, :20], axis=-1)
                         / jnp.maximum(jnp.sum(tr["last_income"][:, -100:, :], axis=-1),
                                       1e-12)))
    att = float(jnp.mean(jnp.sum(tr["listen_influence"][:, -100:, :20], axis=-1)))
    pow_ = float(jnp.mean(jnp.sum(tr["influence"][:, -100:, :20], axis=-1)))
    assert inc < 0.05 and att < 0.05 and pow_ < 0.05


# --- rung 5: the flywheel emerges under coupling (same key, dials on vs off) ------

def test_coupled_composite_below_sealed():
    _, _, coupled = _run(n_steps=300)
    _, _, sealed = _run(n_steps=300, **SEALED)

    def composite(tr):
        inc = jnp.sum(tr["last_income"][:, -75:, :20], axis=-1) / jnp.maximum(
            jnp.sum(tr["last_income"][:, -75:, :], axis=-1), 1e-12)
        att = jnp.sum(tr["listen_influence"][:, -75:, :20], axis=-1)
        pow_ = jnp.sum(tr["influence"][:, -75:, :20], axis=-1)
        return float(jnp.mean((inc + att + pow_) / 3.0))

    assert composite(coupled) < composite(sealed) - 0.01


# --- rung 6: the honest region — no AI arrivals, no disempowerment ----------------

def test_no_arrivals_keeps_human_shares_high():
    _, _, tr = _run(n_steps=200, first_arrival=9999)
    inc = float(jnp.mean(jnp.sum(tr["last_income"][:, -50:, :20], axis=-1)
                         / jnp.maximum(jnp.sum(tr["last_income"][:, -50:, :], axis=-1),
                                       1e-12)))
    pow_ = float(jnp.mean(jnp.sum(tr["influence"][:, -50:, :20], axis=-1)))
    assert inc > 0.9
    assert pow_ > 0.6


# --- rung 7: the polity tracks its median when sealed and unthreatened ------------

def test_sealed_unthreatened_policy_tracks_median():
    _, _, tr = _run(n_steps=200, first_arrival=9999, **SEALED)
    median = float(jnp.median(tr["ideal"][0, 0, :20]))
    late_policy = float(jnp.mean(tr["policy_target"][:, -50:]))
    assert abs(late_policy - median) < 0.15
