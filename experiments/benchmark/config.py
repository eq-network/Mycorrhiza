"""The run spec: (scenario, condition, environment overrides, seeds, horizon).

Schedules live in the scenario's CONDITIONS triples (``scenarios.py``) — the run spec
only picks which scenario/condition to execute and at what scale.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass(frozen=True)
class RunSpec:
    scenario: str = "governed_commons"
    name: str = "baseline"                      # condition, a key of the scenario's CONDITIONS
    env_overrides: Dict[str, Any] = dataclasses.field(default_factory=dict)
    n_seeds: int = 32
    T: int = 200
    seed: int = 0
