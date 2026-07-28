"""
The environment contract: ``EnvSpec``.

An environment in this library is not a class hierarchy — it is a bundle of pure functions over
``GraphState`` plus an evaluation suite. Formalizing that bundle as a single object is what makes
environments *reproducible by construction*: every environment that fills this contract inherits
the same compiled execution, ``vmap``-over-seeds batching, trajectory tracing, and metric wiring
for free, so adding a new one is "implement the five fields, register a name" rather than "wire a
bespoke run loop".

    EnvSpec ::= {
      config   : frozen params (closed over by the transforms, never stored in state)
      init_fn  : key -> GraphState                  (per-seed initial state; vmap-friendly)
      round_fn : (state, t, key) -> state           (one tick; == core.scan.RoundFn)
      trace_fn : state -> dict                       (per-step readout for trajectories)
      metrics  : { name : (trace -> scalar) }        (offline evaluation, e.g. the GovSim suite)
    }

``run`` / ``run_batch`` are thin wrappers over ``core.scan`` — the *only* place execution lives —
so the substrate code never touches the scan plumbing. Governance and measurement compose on top
by wrapping ``round_fn`` and extending ``trace_fn`` / ``metrics``; the contract does not change.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.reduce import Reducer, run_scan_reduce, run_scan_reduce_batch
from cilib.core.scan import run_scan, run_scan_batch, RoundFn, TraceFn


# A metric here scores a *trajectory* (the stacked trace_fn output), not a single state — that is
# how the GovSim suite (survival_time, efficiency, ...) is shaped. Distinct from the inline
# per-step ``metrics/transform.make_metrics_transform``, which writes scalars into the scan carry.
MetricFn = Callable[[Any], Any]


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """A fully-specified, runnable environment. Construct via a registered builder (``make_env``)."""

    name: str
    config: Any
    init_fn: Callable[[Any], GraphState]
    round_fn: RoundFn
    trace_fn: Optional[TraceFn] = None
    metrics: Dict[str, MetricFn] = dataclasses.field(default_factory=dict)

    def run(self, key: Any, n_steps: int,
            trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """One rollout. Returns ``(final_state, trace)``; trace fields have leading axis ``n_steps``.

        ``key`` is split into independent init / run keys (the ``run_scan_batch`` convention) so a
        single ``run`` and one seed of a ``run_batch`` produce the same trajectory for a given key.
        """
        tf = trace_fn if trace_fn is not None else self.trace_fn
        k_init, k_run = jr.split(key)
        return run_scan(self.round_fn, self.init_fn(k_init), n_steps, k_run, trace_fn=tf)

    def run_batch(self, key: Any, n_seeds: int, n_steps: int,
                  trace_fn: Optional[TraceFn] = None) -> Tuple[GraphState, Any]:
        """``vmap``ped multi-seed sweep. Returns ``(final_states, traces)`` with leading axis
        ``(n_seeds,)``; trace fields are shaped ``(n_seeds, n_steps, ...)``. Compiles to one
        ``vmap(scan(...))``."""
        tf = trace_fn if trace_fn is not None else self.trace_fn
        keys = jr.split(key, n_seeds)
        return run_scan_batch(self.round_fn, self.init_fn, n_steps, keys, trace_fn=tf)

    def run_reduced(self, key: Any, n_steps: int, reducers: Dict[str, Reducer],
                    trace_fn: Optional[TraceFn] = None
                    ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """One rollout scoring ``reducers`` INSIDE the scan — the constant-memory path.

        Use this when the stacked trace would not fit: ``metrics`` consume a
        materialized ``(T, N)`` trajectory, which is O(T) larger than the state
        it came from (see ``core/reduce.py`` for the measured ratio). Reducers
        fold in the carry instead, so cost is O(accumulator).

        ``trace_fn`` defaults to ``None`` here rather than to ``self.trace_fn``:
        the environments' default traces are exactly the per-agent arrays you are
        trying not to materialize, so inheriting one would defeat the call. Pass
        a deliberately cheap one — (T,)-shaped scalars are ~8 KB at T=2000 and
        worth keeping for timelines.

        Reducers are passed explicitly rather than carried on the spec because
        ``GameSpec`` is frozen (CLAUDE.md) and could not propagate them through
        ``close()``; an always-empty field would be dead weight. Environments
        expose a ``make_reducers(cfg)`` factory instead.

        Returns:
            ``(final_state, trace, reduced)``.
        """
        k_init, k_run = jr.split(key)
        return run_scan_reduce(self.round_fn, self.init_fn(k_init), n_steps,
                               k_run, reducers, trace_fn=trace_fn)

    def run_reduced_batch(self, key: Any, n_seeds: int, n_steps: int,
                          reducers: Dict[str, Reducer],
                          trace_fn: Optional[TraceFn] = None
                          ) -> Tuple[GraphState, Any, Dict[str, Any]]:
        """``vmap``ped sweep with in-scan reduction. ``reduced[name]`` is shaped
        ``(n_seeds,)`` for a scalar metric — already the per-seed vector the
        benchmark harness bootstraps over, with no trajectory kept."""
        keys = jr.split(key, n_seeds)
        return run_scan_reduce_batch(self.round_fn, self.init_fn, n_steps, keys,
                                     reducers, trace_fn=trace_fn)

    def evaluate(self, trace: Any) -> Dict[str, Any]:
        """Score a trajectory (or batch of trajectories) with every metric in the suite."""
        return {name: fn(trace) for name, fn in self.metrics.items()}
