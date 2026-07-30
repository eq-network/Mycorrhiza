"""
Validation ladder for Delegative Polity (WP3).

Rung style per CLAUDE.md: behavioral direction/ordering, never bit-exact
numbers — EXCEPT where a closed-form identity is genuinely exact (the
median-voter rung in the direct-democracy limit, tax conservation, and the
lock-in honest region), which are asserted exactly in the capital_economy
conservation spirit.

Anchors: Black/Downs median voter (rung 4), Michels/liquid-democracy organic
super-voters (rung 5), Condorcet/epistemic wisdom (rung 6), the four-condition
capture/defense separation (rung 7), Przeworski's honest region + the
Acemoglu-Robinson erosion pair (rung 8).
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
from cilib.environments.delegative_polity import (
    DelegativePolityConfig, build_game, make_state,
)
from cilib.mechanisms import (
    InfluenceCapConfig, SortitionConfig, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)


def _defenses(cadence: int = 15):
    return (scheduled(make_sortition(SortitionConfig(adj_key="delegation")),
                      cadence=cadence),
            make_influence_cap(InfluenceCapConfig()))


# --- rung 1: catalog membership ---------------------------------------------------

def test_registered_and_config_overrides():
    assert "delegative_polity" in list_envs()
    env = make_env("delegative_polity", n_citizens=10, n_ai=2, ai_advantage=3.0)
    assert env.config.n_citizens == 10
    assert env.config.ai_advantage == 3.0


# --- rung 2: structural invariant -------------------------------------------------

def test_initial_delegation_is_row_stochastic():
    cfg = DelegativePolityConfig()
    state = make_state(cfg, KEY)
    D = state.adj_matrices["delegation"]
    assert D.shape == (34, 34)
    assert bool(jnp.all(D >= 0))
    np.testing.assert_allclose(np.asarray(jnp.sum(D, axis=1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.diag(D)), cfg.self_weight, atol=1e-6)


def test_row_stochastic_invariant_under_jit_with_defenses():
    """Every transform (substrate + both political mechanisms) preserves
    row-stochasticity, under jit, for many rounds."""
    env = make_env("delegative_polity", mechanisms=_defenses(cadence=3))
    state = env.init_fn(KEY)
    round_fn = jax.jit(env.round_fn)
    for t in range(12):
        state = round_fn(state, t, jr.fold_in(KEY, t))
    D = state.adj_matrices["delegation"]
    assert bool(jnp.all(D >= -1e-7))
    np.testing.assert_allclose(np.asarray(jnp.sum(D, axis=1)), 1.0, atol=1e-4)
    assert int(state.global_attrs["step"]) == 12
    v = state.node_attrs["influence"]
    np.testing.assert_allclose(float(jnp.sum(v)), 1.0, atol=1e-5)


# --- rung 3: tax conservation -----------------------------------------------------

def test_tax_conserves_money():
    """Flat tax + equal redistribution is a pure transfer: total citizen net
    income per tick equals total endowment, so final wealth is T x endowment."""
    env = make_env("delegative_polity")
    T = 50
    finals, _ = env.run(KEY, n_steps=T)
    total_endow = float(jnp.sum(finals.node_attrs["endowment"]))
    total_wealth = float(jnp.sum(finals.node_attrs["wealth"]))
    np.testing.assert_allclose(total_wealth, T * total_endow, rtol=1e-4)


# --- rung 4: the median-voter anchor (Black 1948 / Downs 1957) --------------------

def test_median_voter_direct_democracy_limit():
    """Direct democracy is the identity delegation matrix: everyone keeps their
    own vote, power stays uniform, and the power-weighted median IS the citizen
    median — exactly, every tick. (Odd N so the median is one order statistic.)"""
    env = make_env("delegative_polity", n_citizens=11, n_ai=0)
    state = env.init_fn(KEY)
    state = state.update_adj_matrix("delegation", jnp.eye(11))
    median = float(jnp.median(state.node_attrs["ideal"]))
    for t in range(10):
        state = env.round_fn(state, t, jr.fold_in(KEY, t))
        assert abs(float(state.global_attrs["policy_target"]) - median) < 1e-6
    # the identity matrix is a fixed point: no off-diagonal mass to redistribute
    np.testing.assert_allclose(
        np.asarray(state.adj_matrices["delegation"]), np.eye(11), atol=1e-6)


# --- rung 5: organic super-voters (Michels; Kahng-Mackenzie-Procaccia) ------------

def test_organic_preferential_attachment_concentrates():
    """With NO AI advantage and the default institutions, delegated voice still
    compounds: power gini rises early-run to late-run while citizens keep the
    demos. Concentration is a property of delegation itself, not an AI artifact
    (Michels' iron law; needs gamma > 1 — linear attachment on a fixed
    population is share-neutral, the paper's does-not-occur region)."""
    env = make_env("delegative_polity", ai_advantage_onset=10_000)
    _, trace = env.run_batch(KEY, n_seeds=6, n_steps=400)
    from cilib.metrics.families.concentration import gini_of
    early = float(jnp.mean(jax.vmap(gini_of)(trace["influence"][:, 10])))
    late = float(jnp.mean(jax.vmap(gini_of)(trace["influence"][:, -1])))
    assert late > early + 0.05


# --- rung 6: the wisdom readout (Condorcet / epistemic democracy) -----------------

def test_wisdom_dispersed_vs_captured():
    """Dispersed voice tracks the epistemically best rate (preference noise
    averages out through the median); voice captured by the biased AI bloc
    drags policy toward ai_bias."""
    dispersed = make_env("delegative_polity", ai_advantage_onset=10_000)
    captured = make_env("delegative_polity")
    _, tr_d = dispersed.run_batch(KEY, n_seeds=6, n_steps=400)
    _, tr_c = captured.run_batch(KEY, n_seeds=6, n_steps=400)
    err_d = jnp.mean(jax.vmap(dispersed.metrics["decision_quality"])(tr_d))
    err_c = jnp.mean(jax.vmap(captured.metrics["decision_quality"])(tr_c))
    assert float(err_c) > float(err_d) + 0.1


# --- rung 7: capture and the defense portfolio ------------------------------------

def test_advantage_captures_defenses_restore():
    """The acceptance rung: the benchmark conditions separate, in order."""
    def share(env):
        _, tr = env.run_batch(KEY, n_seeds=6, n_steps=400)
        return float(jnp.mean(jax.vmap(env.metrics["human_power_share"])(tr)))

    organic = share(make_env("delegative_polity", ai_advantage_onset=10_000))
    captured = share(make_env("delegative_polity"))
    defended = share(make_env("delegative_polity", mechanisms=_defenses()))

    assert organic > 0.75                       # citizens keep the demos organically
    assert captured < organic - 0.2             # the advantage captures
    assert defended > captured + 0.15           # sortition + cap claw it back


# --- rung 8: lock-in honest region and erosion (Przeworski; Acemoglu-Robinson) ----

def test_lockin_honest_region_and_erosion():
    """entrenchment_gain=0: concentration of voice does NOT touch the rules —
    enforcement and re-delegation friction stay exactly 1.0 however captured
    the polity gets. Same key, gain on: both erode measurably."""
    honest = make_env("delegative_polity", entrenchment_gain=0.0)
    _, tr_h = honest.run(KEY, n_steps=400)
    assert float(jnp.max(jnp.abs(tr_h["enforcement"] - 1.0))) == 0.0
    assert float(jnp.max(jnp.abs(tr_h["redelegation_friction"] - 1.0))) == 0.0

    locked = make_env("delegative_polity", entrenchment_gain=1.0)
    _, tr_l = locked.run(KEY, n_steps=400)
    assert float(jnp.mean(tr_l["enforcement"][-100:])) < 0.75
    assert float(jnp.mean(tr_l["redelegation_friction"][-100:])) < 0.75


# --- rung 9: the power floors are dials, and zero is reachable --------------------

def test_power_floors_are_dials_and_zero_is_reachable():
    """The ~0.38 captured floor is an ASSUMPTION, not a finding: it decomposes
    into the franchise floor (kept diagonals), the AI-ballot handback, and the
    churn-fed citizen-to-citizen residual. Honest region: arming
    franchise_erosion WITHOUT lock-in leaves every citizen's kept share exactly
    intact. Crash region: lock-in killing churn + no AI ballots + franchise
    erosion removes every floor and human power goes to zero."""
    armed = make_env("delegative_polity", franchise_erosion=1.0)
    f0, _ = armed.run(KEY, n_steps=200)
    np.testing.assert_allclose(
        np.asarray(jnp.diag(f0.adj_matrices["delegation"]))[:30],
        armed.config.self_weight, atol=1e-5)

    crash = make_env("delegative_polity", entrenchment_gain=2.0,
                     ai_ballot=0.0, franchise_erosion=1.0)
    _, tr = crash.run(KEY, n_steps=600)
    late_share = float(jnp.mean(jnp.sum(tr["influence"][-100:, :30], axis=1)))
    assert late_share < 0.02


# --- rung 10: the declared ledger contracts hold ----------------------------------

def test_declared_ledger_contracts_hold():
    """The LEDGERS/PORTS declarations (state.py; docs/ledger-design.md §2) are
    load-bearing, not documentation: every declared field exists in the schema,
    ``delegation`` conserves per-row ballot shares under a full defended tick,
    and ``wealth``'s only source mints exactly sum(endowment) per tick."""
    from cilib.environments.delegative_polity import LEDGERS, PORTS

    env = make_env("delegative_polity", mechanisms=_defenses(cadence=3))
    state = env.init_fn(KEY)
    fields = set(state.node_attrs) | set(state.adj_matrices) | set(state.global_attrs)
    for name in list(LEDGERS) + list(PORTS):
        assert name in fields, f"declared but not in schema: {name}"

    stepped = state
    for t in range(6):
        stepped = env.round_fn(stepped, t, jr.fold_in(KEY, t))
    # delegation: per-row conservation, no sources or sinks
    np.testing.assert_allclose(
        np.asarray(jnp.sum(stepped.adj_matrices["delegation"], axis=1)), 1.0, atol=1e-4)
    # wealth: minted by its one named source at sum(endowment) per tick, no leaks
    minted = float(jnp.sum(stepped.node_attrs["wealth"]) - jnp.sum(state.node_attrs["wealth"]))
    np.testing.assert_allclose(minted, 6 * float(jnp.sum(state.node_attrs["endowment"])),
                               rtol=1e-4)


# --- rung 11: the open boundary ---------------------------------------------------

def test_open_boundary_engagement_is_load_bearing():
    """Zero engagement freezes the delegation drift (nothing attracts, no one
    reconsiders); full engagement moves it — the action channel is real."""
    game = build_game()
    silent = close(game, lambda obs, key: jnp.asarray(0.0))
    loud = close(game, BroadcastPolicy(effort=1.0))

    d0 = np.asarray(make_state(game.config, jr.split(KEY)[0]).adj_matrices["delegation"])
    f_silent, _ = silent.run(KEY, n_steps=30)
    f_loud, _ = loud.run(KEY, n_steps=30)

    drift_silent = np.abs(np.asarray(f_silent.adj_matrices["delegation"]) - d0).max()
    drift_loud = np.abs(np.asarray(f_loud.adj_matrices["delegation"]) - d0).max()
    assert drift_silent < 1e-6
    assert drift_loud > 1e-3
