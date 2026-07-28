"""
Composable metrics system for Collective Intelligence Library simulations.

Metrics are pure functions (GraphState -> scalar) organized into families
by theoretical lens (economic, governance, graph theory, etc.).

A metrics Transform auto-generated from a dict of metric functions writes
scalars into pre-allocated arrays in global_attrs at each step, compatible
with JAX's lax.scan and vmap.

Three ways to get a number out of a run, cheapest last:

- ``EnvSpec.metrics``: score a materialized trajectory. Simple, and the default
  — but the trajectory is O(T x N), which is ~3000x the state at N >= 500.
- ``make_metrics_transform``: write per-step scalars into pre-allocated
  ``global_attrs`` arrays inside the pipeline. O(T) per metric.
- ``reducers`` + ``EnvSpec.run_reduced``: fold the metric into the scan carry.
  O(1) per metric, no trajectory retained. See ``core/reduce.py``.
"""

from .transform import make_metrics_transform
from .reducers import REDUCERS
from .export import write_trajectory_csv, write_summary_csv
from .families.economic import ECONOMIC_METRICS
from .families.governance import GOVERNANCE_METRICS
from .families.graph import GRAPH_METRICS
from .families.concentration import CONCENTRATION_METRICS
from .families.spectral import SPECTRAL_METRICS

__all__ = [
    'make_metrics_transform',
    'REDUCERS',
    'write_trajectory_csv',
    'write_summary_csv',
    'ECONOMIC_METRICS',
    'GOVERNANCE_METRICS',
    'GRAPH_METRICS',
    'CONCENTRATION_METRICS',
    'SPECTRAL_METRICS',
]
