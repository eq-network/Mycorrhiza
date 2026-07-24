"""Condition resolution: (mechanism key, config, schedule) triples -> Transforms.

The condition dicts themselves live per-scenario in ``scenarios.py``; this module holds
only the resolver. Mechanisms are pure catalog rules; the ``ScheduleSpec`` owns WHEN
they apply (re-vote cadence, onset tick for regime shifts). ``None`` = every tick.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from cilib.core.category import Transform
from cilib.core.schedule import ScheduleSpec, apply_schedule
from cilib.mechanisms import REGISTRY as MECH_REGISTRY


def resolve_mechanisms(
        spec: List[Tuple[str, Any, Optional[ScheduleSpec]]]) -> Tuple[Transform, ...]:
    """Instantiate a condition's portfolio: catalog factory + schedule wrapper."""
    return tuple(apply_schedule(MECH_REGISTRY[name](cfg), sched)
                 for name, cfg, sched in spec)
