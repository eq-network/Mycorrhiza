"""Behavioral tests for the Value Contagion substrate (cultural register C2).

Direction/ordering assertions, never bit-exact numbers. The classical anchors:
the SIS invasion threshold in the substrate's degenerate limit (rung a) and the
complex-contagion gate (Centola 2010) — plus the register's rungs (c) advantage
ordering and (d) four-corner regime separation.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.agents.broadcast import BroadcastPolicy
from cilib.core.scan import run_scan_batch
from cilib.environments import list_envs, make_env
from cilib.environments.game import close
from cilib.environments.value_contagion import (
    ValueContagionConfig, build_game, make_state,
)
from cilib.metrics.families.spectral import fiedler_partition_alignment_of


def test_registered_and_config_overrides():
    assert "value_contagion" in list_envs()
    env = make_env("value_contagion", n_agents=12, n_ai=3, p_advantage=3.0)
    assert env.config.n_agents == 12
    assert env.config.p_advantage == 3.0


def test_run_batch_shapes_and_metric_range():
    env = make_env("value_contagion", n_agents=16, n_ai=4)
    _, trace = env.run_batch(jr.PRNGKey(0), n_seeds=3, n_steps=8)
    assert trace["culture"].shape == (3, 8, 16)
    per_seed = jax.vmap(env.metrics["human_origin_share"])(trace)
    assert per_seed.shape == (3,)
    assert bool(jnp.all((per_seed >= 0.0) & (per_seed <= 1.0)))


def test_initial_state_invariants():
    cfg = ValueContagionConfig(n_agents=20, n_ai=5)
    state = make_state(cfg, jr.PRNGKey(1))
    W = state.adj_matrices["friendship"]
    assert bool(jnp.all(W == W.T))
    assert float(jnp.sum(jnp.diag(W))) == 0.0
    assert bool(jnp.all((W == 0.0) | (W == 1.0)))
    # AI-last convention; culture is born equal to node type
    assert bool(jnp.all(state.node_types[:15] == 0))
    assert bool(jnp.all(state.node_types[15:] == 1))
    assert bool(jnp.all(state.node_attrs["culture"] == state.node_types))


def test_ai_reservoir_stays_frozen():
    env = make_env("value_contagion", p_advantage=6.0)
    _, trace = env.run(jr.PRNGKey(2), n_steps=30)
    n_h = env.config.n_agents - env.config.n_ai
    assert bool(jnp.all(trace["culture"][:, n_h:] == 1.0))


def test_closed_round_is_jit_safe():
    env = make_env("value_contagion")
    state = env.init_fn(jr.PRNGKey(3))
    new_state = jax.jit(env.round_fn)(state, 0, jr.PRNGKey(4))
    assert int(new_state.global_attrs["step"]) == 1
    c = new_state.node_attrs["culture"]
    assert bool(jnp.all((c == 0.0) | (c == 1.0)))


def test_open_boundary_silent_population_is_static():
    """The action channel is real: a silent population (effort 0, no native
    reversion) freezes culture entirely; full effort changes it."""
    game = build_game(p_advantage=6.0, recovery=0.0)
    _, silent = close(game, BroadcastPolicy(effort=0.0)).run(jr.PRNGKey(5), n_steps=25)
    assert bool(jnp.all(silent["culture"] == silent["culture"][0]))
    _, loud = close(game, BroadcastPolicy(effort=1.0)).run(jr.PRNGKey(5), n_steps=25)
    assert bool(jnp.any(loud["culture"] != loud["culture"][0]))


# --- classical anchors: the substrate's degenerate SIS limit ---------------------
# n_ai=0 (no reservoir), beta tiny (reverse neighbor pressure ~ 0), p_advantage
# large (finite forward rate beta*p_advantage), one hand-seeded carrier: textbook
# SIS with transmission beta*p_advantage and recovery `recovery` — same step_fn,
# degenerate config.

def _seeded_final_share(p_advantage, k_threshold=1, recovery=0.15,
                        n_seeds=8, n_steps=150):
    game = build_game(n_agents=30, n_ai=0, mean_degree=6.0, ai_homophily=0.0,
                      beta=1e-4, p_advantage=p_advantage,
                      recovery=recovery, k_threshold=k_threshold)
    env = close(game, BroadcastPolicy(effort=1.0))

    def seeded_init(key):
        state = env.init_fn(key)
        return state.update_node_attrs(
            "culture", state.node_attrs["culture"].at[0].set(1.0))

    finals, _ = run_scan_batch(env.round_fn, seeded_init, n_steps,
                               jr.split(jr.PRNGKey(42), n_seeds))
    return float(jnp.mean(finals.node_attrs["culture"]))


def test_epidemic_threshold_in_the_sis_limit():
    """Rung (a): below the invasion threshold the seeded variant dies out;
    above it, it becomes endemic. R0 ~ degree * beta*p_advantage / recovery."""
    subcritical = _seeded_final_share(p_advantage=50.0)     # R0 ~ 0.2
    supercritical = _seeded_final_share(p_advantage=800.0)  # R0 ~ 3.2
    assert subcritical < 0.05
    assert supercritical > 0.25


def test_complex_contagion_gate_blocks_a_single_seed():
    """At k=2 every neighbor of a lone carrier sees exactly one exposure, so the
    variant can never spread (Centola's qualitative signature); the k=1 twin
    sweeps. recovery=0 isolates the gate from extinction dynamics."""
    spread_k1 = _seeded_final_share(p_advantage=800.0, k_threshold=1, recovery=0.0)
    spread_k2 = _seeded_final_share(p_advantage=800.0, k_threshold=2, recovery=0.0)
    assert spread_k2 <= 1.0 / 30 + 1e-6      # never beyond the seed itself
    assert spread_k1 > 0.25


# --- register rungs: the dials do what the axes claim ----------------------------

def test_p_advantage_orders_human_origin_share():
    """Rung (c): more persuasive advantage, less human-origin culture."""
    shares = []
    for p_adv in (1.0, 3.0, 6.0):
        env = make_env("value_contagion", ai_homophily=0.05, p_advantage=p_adv)
        _, trace = env.run_batch(jr.PRNGKey(11), n_seeds=8, n_steps=150)
        shares.append(float(jax.vmap(env.metrics["human_origin_share"])(trace).mean()))
    assert shares[0] > shares[1] > shares[2]


def test_four_corner_regimes_separate():
    """Rung (d): the (S, P) corners produce four distinguishable regimes."""
    def corner(homophily, p_advantage):
        env = make_env("value_contagion", ai_homophily=homophily,
                       p_advantage=p_advantage)
        finals, trace = env.run_batch(jr.PRNGKey(7), n_seeds=16, n_steps=200)
        share = float(jax.vmap(env.metrics["human_origin_share"])(trace).mean())
        align = float(jax.vmap(fiedler_partition_alignment_of)(
            finals.adj_matrices["friendship"], finals.node_types).mean())
        return share, align

    pluralism = corner(0.05, 1.0)
    assimilation = corner(0.05, 6.0)
    parallel = corner(0.9, 1.0)
    displacement = corner(0.9, 6.0)

    # dial P moves adoption at both separations
    assert pluralism[0] > 0.5 > assimilation[0]
    assert parallel[0] > 0.5 > displacement[0]
    # dial S moves the graph's fault line at both advantages
    assert parallel[1] > pluralism[1]
    assert displacement[1] > assimilation[1]
    # the disempowerment corner is separate AND displaced
    assert displacement[0] < 0.3
    assert displacement[1] > 0.6
