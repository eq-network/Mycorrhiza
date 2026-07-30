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
               ("consume_spend", "invest_spend", "broadcast_spend", "lobby_spend"))
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
