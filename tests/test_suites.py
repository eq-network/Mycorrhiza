"""Behavioral contract for suites: every member of every suite resolves in the
environments catalog and its factory builds a runnable EnvSpec."""
import jax.random as jr

from cilib.environments import REGISTRY, make_env
from cilib.environments.suites import SUITES, GRADUAL_DISEMPOWERMENT


def test_suite_members_resolve_in_registry():
    for suite_name, suite in SUITES.items():
        for role, env_name in suite["members"].items():
            assert env_name in REGISTRY, (
                f"suite {suite_name!r} member {role!r} -> {env_name!r} not in REGISTRY")


def test_gd_v01_membership_is_the_wp_series():
    assert GRADUAL_DISEMPOWERMENT["version"] == "0.1"
    assert GRADUAL_DISEMPOWERMENT["members"] == {
        "economy": "capital_economy",
        "culture": "influence_exchange",
        "politics": "delegative_polity",
        "coupled": "coupled_society",
    }


def test_gd_v01_members_build_and_step():
    key = jr.PRNGKey(0)
    for env_name in GRADUAL_DISEMPOWERMENT["members"].values():
        env = make_env(env_name)
        finals, traces = env.run_batch(key, n_seeds=2, n_steps=3)
        assert traces is not None, env_name
