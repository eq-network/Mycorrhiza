"""
SystemGraph — the pipeline's dependency DAG read as communication.

The lab page's System view (eq-network ``prototypes/playground.html`` §4b)
draws every game as one information-passing graph: agents, stocks, and the
mechanisms themselves are nodes, and every within-tick interaction is a
message on an edge. That object is already latent here: every ``@transform``
declares reads/writes, so a mechanism's in-edges ARE its reads and its
out-edges ARE its writes — ``compile_pipeline`` orders exactly these edges as
computation; this module emits the same edges as communication. A spec-defined
scenario therefore gets a faithful system graph nobody hand-drew, and a future
graph *editor*'s save file and the visualization's input are the same JSON.

Two node kinds:

- ``field``: an evolving state array (``node_attrs`` / ``adj_matrices`` /
  ``edge_attrs`` entries, and jnp-array ``global_attrs`` — static Python aux is
  config, not a message carrier). ``rng_key``/``step`` are tagged
  ``bookkeeping`` so renderers can dim the plumbing without hiding it.
- ``transform``: one pipeline step (substrate step, mechanism, counter), with
  its declared reads/writes inline. Toggling a mechanism literally adds or
  removes its node — the System view's claim, at the data level.

v1 scope is the *declared* pipeline only: the GameSpec boundary (observe →
policy → actions) carries no ``@transform`` metadata, so the closing policy is
not derived here — the trajectory's action fields are the visible residue of
that boundary. Consumed by ``examples/05_export_trajectory.py`` per
``docs/web-trajectory-contract.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cilib.core.graph import GraphState

_BOOKKEEPING = frozenset({"rng_key", "step"})


def system_graph(steps: Sequence[Any], state: GraphState) -> Dict[str, Any]:
    """Derive ``{nodes, edges}`` from transform metadata + the state schema.

    ``steps`` is the same list ``compile_pipeline`` receives (an environment's
    ``build_steps(cfg, mechanisms)``); ``state`` any state built by its
    ``make_state`` (only the schema is read, never the values).
    """
    nodes: List[Dict[str, Any]] = []
    field_ids = set()

    def add_field(name, family, shape):
        nodes.append({"id": name, "kind": "field", "family": family,
                      "shape": list(shape),
                      **({"bookkeeping": True} if name in _BOOKKEEPING else {})})
        field_ids.add(name)

    for family, mapping in (("node_attrs", state.node_attrs),
                            ("adj_matrices", state.adj_matrices),
                            ("edge_attrs", state.edge_attrs)):
        for name, arr in mapping.items():
            add_field(name, family, arr.shape)
    for name, val in state.global_attrs.items():
        if hasattr(val, "shape"):          # jnp arrays evolve; Python aux is static
            add_field(name, "global_attrs", val.shape)

    edges: List[Dict[str, str]] = []
    seen: Dict[str, int] = {}
    for i, t in enumerate(steps):
        base = getattr(t, "name", None) or getattr(t, "__name__", f"step_{i}")
        seen[base] = seen.get(base, 0) + 1
        tid = base if seen[base] == 1 and base not in field_ids else f"{base}#{seen[base]}"
        reads = sorted(getattr(t, "reads", frozenset()))
        writes = sorted(getattr(t, "writes", frozenset()))
        nodes.append({"id": tid, "kind": "transform", "reads": reads, "writes": writes})
        edges.extend({"from": r, "to": tid} for r in reads)
        edges.extend({"from": tid, "to": w} for w in writes)

    return {"nodes": nodes, "edges": edges}
