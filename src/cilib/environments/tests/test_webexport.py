"""Behavioral contract for webexport.trajectory_payload — the single producer of
the web trajectory payload (docs/web-trajectory-contract.md v1.1).

The parity rung pins the mapping byte-for-byte against an inline naive
implementation of the contract's rules on a real run, so refactors of the
producer cannot silently change what ships to the page.
"""
import json

import jax.random as jr
import numpy as np
import pytest

from cilib.environments import make_env
from cilib.environments.webexport import trajectory_payload


def _naive_payload(trace, finals, game_id, n_steps, seed, params, scalars):
    """The contract's mapping rules, written independently (mirrors the
    pre-refactor examples/05 body)."""
    payload = {"global": {}, "node": {}, "static": {}}
    n_agents = None
    for name, arr in trace.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            payload["global"][name] = arr.tolist()
        elif arr.ndim == 2:
            n_agents = arr.shape[1]
            if bool(np.all(arr == arr[0])):
                payload["static"][name] = arr[0].tolist()
            else:
                payload["node"][name] = arr.ravel().tolist()
    if finals.adj_matrices:
        payload["adj"] = {
            name: np.asarray(arr.todense() if hasattr(arr, "todense") else arr)
                    .ravel().tolist()
            for name, arr in finals.adj_matrices.items()}
    payload["meta"] = {"gameId": game_id, "T": n_steps, "N": n_agents,
                       "seed": seed, "params": dict(params),
                       "scalars": {k: float(v) for k, v in scalars.items()}}
    return payload


def test_parity_byte_identical_on_real_run():
    env = make_env("governed_commons")
    finals, trace = env.run(jr.PRNGKey(0), 5)
    scalars = env.evaluate(trace)
    params = {"condition": "baseline"}
    ours = trajectory_payload(trace, finals, game_id="governed_commons",
                              n_steps=5, seed=0, params=params, scalars=scalars)
    naive = _naive_payload(trace, finals, "governed_commons", 5, 0, params, scalars)
    assert json.dumps(ours, sort_keys=True) == json.dumps(naive, sort_keys=True)


def test_mapping_rules_and_producer_options():
    T, N = 4, 3
    trace = {
        "scalar_series": np.arange(T, dtype=np.float32),
        "varying": np.arange(T * N, dtype=np.float32).reshape(T, N),
        "constant": np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (T, 1)),
    }
    p = trajectory_payload(trace, None, game_id="g", n_steps=T, seed=1,
                           params={}, scalars={})
    assert list(p["global"]) == ["scalar_series"]
    assert list(p["static"]) == ["constant"]
    assert p["node"]["varying"] == list(range(T * N))     # row-major t*N+i
    assert p["meta"]["N"] == N and "adj" not in p and "system" not in p

    # whitelist drops fields but N is still inferred from the full trace
    p2 = trajectory_payload(trace, None, game_id="g", n_steps=T, seed=1,
                            params={}, scalars={}, whitelist=["scalar_series"])
    assert list(p2["global"]) == ["scalar_series"]
    assert not p2["node"] and not p2["static"] and p2["meta"]["N"] == N

    # derived (T,) series land in global; rounding applies everywhere
    derived = {"share": np.array([0.123456, 0.2, 0.3, 0.4])}
    p3 = trajectory_payload(trace, None, game_id="g", n_steps=T, seed=1,
                            params={}, scalars={}, derived_global=derived,
                            round_decimals=4)
    assert p3["global"]["share"][0] == pytest.approx(0.1235)

    with pytest.raises(ValueError):
        trajectory_payload({"bad": np.zeros((2, 2, 2))}, None, game_id="g",
                           n_steps=2, seed=0, params={}, scalars={})
    with pytest.raises(ValueError):
        trajectory_payload(trace, None, game_id="g", n_steps=T, seed=0,
                           params={}, scalars={},
                           derived_global={"bad": np.zeros((2, 2))})
