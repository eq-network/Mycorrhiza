"""
webexport — the single producer of the web trajectory payload
(docs/web-trajectory-contract.md, v1.1).

The generic trace→payload mapping used by every file producer
(`examples/05_export_trajectory.py`, `experiments/gd_bundles/export.py`):

    (T,) fields -> "global";  (T, N) varying -> "node" (row-major t*N+i);
    (T, N) constant -> "static"

Producer options, all within contract v1.1 (see the contract's Producers note):
``derived_global`` appends engine-computed (T,) reductions as extra global
fields; ``whitelist`` restricts which trace fields ship; ``round_decimals``
rounds before serialization (mandatory for sweep bundles — float32 ``tolist()``
reprs otherwise triple payload size); ``include_adj=False`` omits the optional
``adj`` section. Defaults reproduce the historical 05 output byte-for-byte
(pinned by ``tests/test_webexport.py``).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def _series(arr, round_decimals: Optional[int]):
    if round_decimals is not None:
        # float64 BEFORE rounding: rounding float32 then .tolist() yields doubles
        # with 17-digit reprs (0.029400000450015068) — tripling payload size
        arr = np.round(np.asarray(arr, dtype=np.float64), round_decimals)
    return arr.tolist()


def trajectory_payload(trace: Mapping[str, object], finals=None, *,
                       game_id: str, n_steps: int, seed: int,
                       params: Mapping[str, object],
                       scalars: Mapping[str, float],
                       system: Optional[dict] = None,
                       whitelist: Optional[Sequence[str]] = None,
                       derived_global: Optional[Mapping[str, object]] = None,
                       round_decimals: Optional[int] = None,
                       include_adj: bool = True) -> dict:
    """Map one run's trace onto the contract payload. No per-environment code.

    ``trace``: dict of (T,) / (T, N) arrays for a SINGLE run. ``finals`` is the
    final GraphState (only its ``adj_matrices`` are read, and only when
    ``include_adj``). Section order (global, node, static, adj?, system?, meta)
    is part of the byte-stability contract with the historical exporter.
    """
    keep = set(whitelist) if whitelist is not None else None
    payload = {"global": {}, "node": {}, "static": {}}
    n_agents = None
    for name, arr in trace.items():
        arr = np.asarray(arr)
        if arr.ndim == 2:
            n_agents = arr.shape[1]
        if keep is not None and name not in keep:
            continue
        if arr.ndim == 1:
            payload["global"][name] = _series(arr, round_decimals)
        elif arr.ndim == 2:
            if bool(np.all(arr == arr[0])):
                payload["static"][name] = _series(arr[0], round_decimals)
            else:
                payload["node"][name] = _series(arr.ravel(), round_decimals)
        else:
            raise ValueError(f"trace field {name!r} has unsupported shape {arr.shape}")

    for name, arr in (derived_global or {}).items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"derived field {name!r} must be (T,), got {arr.shape}")
        payload["global"][name] = _series(arr, round_decimals)

    if include_adj and finals is not None and finals.adj_matrices:
        # densify at the boundary: the wire contract is a flat row-major N*N
        # array whatever the in-engine representation (off the hot path).
        payload["adj"] = {
            name: np.asarray(arr.todense() if hasattr(arr, "todense") else arr)
                    .ravel().tolist()
            for name, arr in finals.adj_matrices.items()}

    if system is not None:
        payload["system"] = system

    payload["meta"] = {
        "gameId": game_id,
        "T": n_steps,
        "N": n_agents,
        "seed": seed,
        "params": dict(params),
        "scalars": {k: float(v) for k, v in scalars.items()},
    }
    return payload
