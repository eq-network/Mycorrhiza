"""Behavioral tests for the SystemGraph emitter (pipeline DAG as communication)."""
import jax.random as jr

from cilib.core.schedule import ScheduleSpec, apply_schedule
from cilib.environments import governed_commons, value_contagion
from cilib.environments.system_graph import system_graph
from cilib.mechanisms import REGISTRY as MECHANISMS, QuotaVoteConfig


def _graph(mod, mechanisms=(), **cfg):
    config = mod.build_game(mechanisms, **cfg).config
    steps = mod.build_steps(config, tuple(mechanisms))
    state = mod.make_state(config, jr.PRNGKey(0))
    return system_graph(steps, state)


def _by_id(g):
    return {n["id"]: n for n in g["nodes"]}


def test_value_contagion_adopt_edges_match_declared_metadata():
    g = _graph(value_contagion)
    nodes = _by_id(g)
    assert nodes["friendship"]["family"] == "adj_matrices"
    assert nodes["adopt"]["kind"] == "transform"
    assert {"from": "friendship", "to": "adopt"} in g["edges"]
    assert {"from": "adopt", "to": "culture"} in g["edges"]
    assert nodes["rng_key"].get("bookkeeping") is True


def test_every_edge_endpoint_is_a_node():
    for mod in (value_contagion, governed_commons):
        g = _graph(mod)
        ids = set(_by_id(g))
        for e in g["edges"]:
            assert e["from"] in ids and e["to"] in ids


def test_toggling_a_mechanism_adds_its_node():
    bare = _graph(governed_commons)
    mech = apply_schedule(MECHANISMS["quota_vote"](QuotaVoteConfig()),
                          ScheduleSpec(cadence=5))
    defended = _graph(governed_commons, mechanisms=(mech,))
    extra = set(_by_id(defended)) - set(_by_id(bare))
    assert len(extra) == 1
    (mech_id,) = extra
    assert "quota_vote" in mech_id
    # the mechanism's out-edge is the enacted policy — the System view's flow
    assert {"from": mech_id, "to": "policy_target"} in defended["edges"]
