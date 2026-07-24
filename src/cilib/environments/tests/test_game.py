"""Behavioral tests for the GameSpec boundary contract (game.py)."""
import jax.numpy as jnp
import jax.random as jr
import pytest

from cilib.core.category import transform
from cilib.core.graph import GraphState
from cilib.environments import GameSpec, close, validate_reads


def _toy_game(n=4):
    """Minimal open game: obs = own score; step adds the action to the score."""
    def init_fn(key):
        return GraphState(
            node_types=jnp.zeros(n, dtype=jnp.int32),
            node_attrs={"score": jnp.zeros(n, dtype=jnp.float32),
                        "last_reward": jnp.zeros(n, dtype=jnp.float32)},
            adj_matrices={}, edge_attrs={},
            global_attrs={"step": jnp.array(0, dtype=jnp.int32)},
        )

    def step_fn(state, actions, key):
        state = state.update_node_attrs("score", state.node_attrs["score"] + actions)
        return state.update_global_attr("step", state.global_attrs["step"] + 1)

    return GameSpec(name="toy", config=None, init_fn=init_fn,
                    observe_fn=lambda s: s.node_attrs["score"], step_fn=step_fn)


def test_close_runs_policy_at_the_boundary():
    game = _toy_game()
    env = close(game, lambda obs, key: obs + 1.0)     # act = observed score + 1
    final, _ = env.run(jr.PRNGKey(0), n_steps=3)
    # scores: 0 -> 1 -> 3 -> 7 (each round adds score+1)
    assert jnp.allclose(final.node_attrs["score"], 7.0)
    assert int(final.global_attrs["step"]) == 3


def test_close_of_same_game_with_different_policies_diverges():
    game = _toy_game()
    passive = close(game, lambda obs, key: jnp.zeros_like(obs))
    active = close(game, lambda obs, key: jnp.ones_like(obs))
    f_passive, _ = passive.run(jr.PRNGKey(0), n_steps=5)
    f_active, _ = active.run(jr.PRNGKey(0), n_steps=5)
    assert float(f_passive.node_attrs["score"].sum()) == 0.0
    assert float(f_active.node_attrs["score"].sum()) > 0.0


def test_rewards_are_a_view_of_state():
    game = _toy_game()
    state = game.init_fn(jr.PRNGKey(0))
    state = state.update_node_attrs("last_reward", jnp.arange(4, dtype=jnp.float32))
    assert jnp.allclose(game.rewards(state), jnp.arange(4, dtype=jnp.float32))


def test_validate_reads_catches_contract_mismatch():
    game = _toy_game()
    state = game.init_fn(jr.PRNGKey(0))

    @transform(reads=["no_such_field"], writes=["score"])
    def broken(s):
        return s

    issues = validate_reads([broken], state)
    assert len(issues) == 1 and "no_such_field" in issues[0]


def test_validate_reads_accepts_reads_satisfied_by_earlier_writes():
    game = _toy_game()
    state = game.init_fn(jr.PRNGKey(0))

    @transform(reads=["score"], writes=["derived"])
    def writes_derived(s):
        return s

    @transform(reads=["derived"], writes=["score"])
    def reads_derived(s):
        return s

    assert validate_reads([writes_derived, reads_derived], state) == []


def test_governed_commons_build_game_rejects_bad_mechanism():
    from cilib.environments.governed_commons import build_game

    @transform(reads=["nonexistent_signal"], writes=["policy_target"])
    def bad_mechanism(s):
        return s

    with pytest.raises(ValueError, match="nonexistent_signal"):
        build_game(mechanisms=(bad_mechanism,))
