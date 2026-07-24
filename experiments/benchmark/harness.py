"""Run conditions, score them against the undefended baseline, emit the scorecard dict.

Scoring reuses ``EnvSpec.metrics`` vmapped over the seed axis — no bespoke scoring.
Two influence readings per condition, both normalized against the undefended baseline
from the SAME seeds:

- **exercised influence** (headline): causal, from paired same-key counterfactual
  rollouts. Each scenario defines its own instrument in ``scenarios.py`` (the commons
  shifts asks from t=0; the economy intervenes MID-RUN, after AI capital entrenches).
- Whatever descriptive metrics the scenario's ``EnvSpec.metrics`` provide, with
  bootstrap CIs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import jax
import jax.random as jr
import numpy as np

from cilib.environments import make_env
from cilib.lab.analysis.bootstrap import bootstrap_ci
from cilib.lab.analysis.influence import influence_preserved

from .config import RunSpec
from .conditions import resolve_mechanisms
from .scenarios import SCENARIOS, ScenarioSpec


def run_condition(spec: RunSpec):
    """Resolve the condition's portfolio, build the env, run the seed batch."""
    scenario = SCENARIOS[spec.scenario]
    mechanisms = resolve_mechanisms(scenario.conditions[spec.name])
    env = make_env(scenario.env_name, mechanisms=mechanisms, **spec.env_overrides)
    finals, trace = env.run_batch(jr.PRNGKey(spec.seed), spec.n_seeds, spec.T)
    return env, trace


def exercised_per_seed(spec: RunSpec) -> np.ndarray:
    """(S,) causal influence for this condition, per the scenario's own instrument."""
    scenario = SCENARIOS[spec.scenario]
    mechanisms = resolve_mechanisms(scenario.conditions[spec.name])
    return scenario.influence_fn(mechanisms, spec)


def per_seed_metrics(env, trace) -> Dict[str, np.ndarray]:
    """{metric name: (S,) per-seed values} via vmap over the leading seed axis."""
    return {name: np.asarray(jax.vmap(fn)(trace)) for name, fn in env.metrics.items()}


def score_condition(name: str, per_seed: Dict[str, np.ndarray],
                    exercised: np.ndarray, baseline_exercised: np.ndarray) -> dict:
    """One scorecard row: the causal influence scores + every env metric with a CI."""
    metrics_ci = {}
    for metric, values in per_seed.items():
        point, lo, hi = bootstrap_ci(values)
        metrics_ci[metric] = {"point": point, "lo": lo, "hi": hi}
    e_point, e_lo, e_hi = bootstrap_ci(exercised)
    metrics_ci["exercised_influence"] = {"point": e_point, "lo": e_lo, "hi": e_hi}
    return {
        "condition": name,
        "influence_preserved": influence_preserved(exercised, baseline_exercised),
        "exercised_influence": e_point,
        "metrics": metrics_ci,
    }


def run_benchmark(scenario: str = "governed_commons",
                  condition_names: Optional[List[str]] = None,
                  n_seeds: int = 32, T: Optional[int] = None, seed: int = 0,
                  env_overrides: Optional[dict] = None) -> dict:
    """Run the baseline plus every requested condition; return the scorecard dict."""
    scn: ScenarioSpec = SCENARIOS[scenario]
    names = list(condition_names) if condition_names else list(scn.conditions)
    horizon = T if T is not None else scn.default_T
    overrides = env_overrides or {}

    def spec(name):
        return RunSpec(scenario=scenario, name=name, env_overrides=overrides,
                       n_seeds=n_seeds, T=horizon, seed=seed)

    baseline_env, baseline_trace = run_condition(spec("baseline"))
    baseline_per_seed = per_seed_metrics(baseline_env, baseline_trace)
    baseline_exercised = exercised_per_seed(spec("baseline"))

    rows = []
    for name in names:
        if name == "baseline":
            per_seed, exercised = baseline_per_seed, baseline_exercised
        else:
            env, trace = run_condition(spec(name))
            per_seed = per_seed_metrics(env, trace)
            exercised = exercised_per_seed(spec(name))
        rows.append(score_condition(name, per_seed, exercised, baseline_exercised))

    return {
        "scenario": scenario,
        "meta": {"n_seeds": n_seeds, "T": horizon, "seed": seed,
                 "env_overrides": overrides},
        "rows": rows,
    }
