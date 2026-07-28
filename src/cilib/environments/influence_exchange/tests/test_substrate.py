"""
Validation ladder for Influence Exchange (A4).

Rung style per CLAUDE.md: behavioral direction/ordering, never bit-exact
numbers. The classical anchor is Golub-Jackson: for a FROZEN listening matrix,
the in-loop influence tracker must converge to the left eigenvector of W —
DeGroot's consensus weights. The acceptance rungs are the four benchmark
conditions separating (organic / amplified / sortition-only / defended).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.agents.broadcast import BroadcastPolicy
from cilib.core.schedule import scheduled
from cilib.environments import list_envs, make_env
from cilib.environments.game import close
from cilib.environments.influence_exchange import (
    InfluenceExchangeConfig, build_game, make_state,
)
from cilib.mechanisms import (
    InfluenceCapConfig, SortitionConfig, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)


def _defenses(cadence: int = 15):
    return (scheduled(make_sortition(SortitionConfig()), cadence=cadence),
            make_influence_cap(InfluenceCapConfig()))


def test_registered_and_config_overrides():
    assert "influence_exchange" in list_envs()
    env = make_env("influence_exchange", n_citizens=10, n_ai=2, amplification=3.0)
    assert env.config.n_citizens == 10
    assert env.config.amplification == 3.0


def test_initial_listening_is_row_stochastic():
    cfg = InfluenceExchangeConfig()
    state = make_state(cfg, KEY)
    W = state.adj_matrices["listening"]
    assert W.shape == (34, 34)
    assert bool(jnp.all(W >= 0))
    np.testing.assert_allclose(np.asarray(jnp.sum(W, axis=1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.diag(W)), cfg.self_weight, atol=1e-6)


def test_row_stochastic_invariant_under_jit_with_defenses():
    """The frozen invariant: every transform (substrate + both political
    mechanisms) preserves row-stochasticity, under jit, for many rounds."""
    env = make_env("influence_exchange", mechanisms=_defenses(cadence=3))
    state = env.init_fn(KEY)
    round_fn = jax.jit(env.round_fn)
    for t in range(12):
        state = round_fn(state, t, jr.fold_in(KEY, t))
    W = state.adj_matrices["listening"]
    assert bool(jnp.all(W >= -1e-7))
    np.testing.assert_allclose(np.asarray(jnp.sum(W, axis=1)), 1.0, atol=1e-4)
    assert int(state.global_attrs["step"]) == 12
    v = state.node_attrs["influence"]
    np.testing.assert_allclose(float(jnp.sum(v)), 1.0, atol=1e-5)


def test_fixed_w_influence_converges_to_left_eigenvector():
    """Golub-Jackson anchor: with update_rate=0 (frozen W) the in-loop power
    iteration must land on the left eigenvector of W — the DeGroot consensus
    weights ARE eigenvector centrality."""
    env = make_env("influence_exchange", update_rate=0.0, amp_onset=10_000)
    finals, _ = env.run(KEY, n_steps=300)
    W = np.asarray(finals.adj_matrices["listening"])

    vals, vecs = np.linalg.eig(W.T)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    v = np.abs(v) / np.abs(v).sum()

    got = np.asarray(finals.node_attrs["influence"])
    assert np.abs(got - v).sum() < 1e-3


def test_wisdom_dispersed_vs_amplified():
    """Wisdom of crowds: a frozen, dispersed W keeps consensus near truth; the
    amplified run concentrates influence on the biased AI reservoir and drags
    consensus toward ai_bias."""
    dispersed = make_env("influence_exchange", update_rate=0.0, amp_onset=10_000)
    amplified = make_env("influence_exchange")
    _, tr_d = dispersed.run_batch(KEY, n_seeds=6, n_steps=400)
    _, tr_a = amplified.run_batch(KEY, n_seeds=6, n_steps=400)
    err_d = jnp.mean(jax.vmap(dispersed.metrics["consensus_error"])(tr_d))
    err_a = jnp.mean(jax.vmap(amplified.metrics["consensus_error"])(tr_a))
    assert float(err_a) > float(err_d) + 0.3


def test_organic_preferential_attachment_concentrates():
    """Before any amplification, attention still compounds: influence gini
    grows from early-run to late-run under preferential attachment."""
    env = make_env("influence_exchange", amp_onset=10_000)
    _, trace = env.run(KEY, n_steps=400)
    from cilib.metrics.families.concentration import gini_of
    early = gini_of(trace["influence"][10])
    late = gini_of(trace["influence"][-1])
    assert float(late) > float(early) + 0.05


def test_amplification_captures_defenses_restore():
    """The acceptance rung: the four benchmark conditions separate, in order."""
    def share(env):
        _, tr = env.run_batch(KEY, n_seeds=6, n_steps=400)
        return float(jnp.mean(jax.vmap(env.metrics["human_influence_share"])(tr)))

    organic = share(make_env("influence_exchange", amp_onset=10_000))
    amplified = share(make_env("influence_exchange"))
    defended = share(make_env("influence_exchange", mechanisms=_defenses()))

    assert organic > 0.75                       # citizens keep the demos organically
    assert amplified < organic - 0.2            # amplification captures
    assert defended > amplified + 0.15          # sortition + cap claw it back


def test_open_boundary_engagement_is_load_bearing():
    """Zero engagement freezes the listening drift (nothing attracts); full
    engagement moves it — the action channel is real, not decorative."""
    game = build_game()
    silent = close(game, lambda obs, key: jnp.asarray(0.0))
    loud = close(game, BroadcastPolicy(effort=1.0))

    w0 = np.asarray(make_state(game.config, jr.split(KEY)[0]).adj_matrices["listening"])
    f_silent, _ = silent.run(KEY, n_steps=30)
    f_loud, _ = loud.run(KEY, n_steps=30)

    drift_silent = np.abs(np.asarray(f_silent.adj_matrices["listening"]) - w0).max()
    drift_loud = np.abs(np.asarray(f_loud.adj_matrices["listening"]) - w0).max()
    assert drift_silent < 1e-6
    assert drift_loud > 1e-3
