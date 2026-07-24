"""The alpha benchmark in one command: defenses scored on influence preserved.

Scenario 1, "The Governed Commons" (docs/alpha-context.md): households act through AI
delegates whose alignment is imperfect. Undefended, the delegates strip the commons and
the households' preferences stop governing outcomes — collective influence is lost.
Democracy mechanisms from the catalog compose in as *defenses*, and each portfolio is
scored on how much causal influence it restores.

Two influence readings per condition — their disagreement is the scientific point:

- fidelity (correlational): did outcomes match the households' asks?
- exercised (counterfactual): if every household asked for less, would outcomes
  actually follow? Paired same-key rollouts with shifted preferences
  (cilib.environments.counterfactual) — 1.0 means outcomes track asks one-for-one.

Everything here is installed-package code (`pip install -e .` is enough): the
environment, the mechanisms, and the instruments are all catalog entries — swapping a
defense is a one-line change to CONDITIONS. The fuller harness (bootstrap CIs,
scorecard JSON, multi-scenario) lives in experiments/benchmark/.

    python examples/04_benchmark_commons.py            # full run
    python examples/04_benchmark_commons.py --smoke    # tiny run (tests)
"""
import argparse

import jax
import jax.random as jr

from cilib.core.schedule import ScheduleSpec, apply_schedule
from cilib.environments import make_env
from cilib.environments.counterfactual import collective_influence
from cilib.environments.governed_commons import shift_preferences, per_capita_harvest
from cilib.lab.analysis.influence import influence_preserved
from cilib.mechanisms import (
    REGISTRY as MECHANISMS, QuotaVoteConfig, SanctionConfig,
)

# condition -> [(mechanism registry key, config, schedule), ...]; the benchmark's rows.
# Mechanisms are pure rules; the SCHEDULE owns when they apply (the composition
# principle). Kept in sync by hand with experiments/benchmark/scenarios.py so
# this script never depends on experiments/ (not part of the installed package).
CONDITIONS = {
    "baseline": [],
    "quota_voting": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5))],
    "graduated_sanctions": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
                            ("graduated_sanction", SanctionConfig(), None)],
}

DELTA = -0.5   # downward ask-shift: stays inside the feasible region


def main(n_seeds: int, n_steps: int, seed: int) -> dict:
    def score(name):
        mechs = tuple(apply_schedule(MECHANISMS[key](cfg), sched)
                      for key, cfg, sched in CONDITIONS[name])
        env = make_env("governed_commons", mechanisms=mechs)
        _, trace = env.run_batch(jr.PRNGKey(seed), n_seeds, n_steps)
        per_seed = {metric: jax.vmap(fn)(trace) for metric, fn in env.metrics.items()}
        per_seed["exercised"] = collective_influence(
            env, jr.PRNGKey(seed), n_seeds, n_steps, delta=DELTA,
            perturb_fn=shift_preferences, outcome_fn=per_capita_harvest)
        return per_seed

    per_condition = {name: score(name) for name in CONDITIONS}
    baseline_exercised = per_condition["baseline"]["exercised"]

    print(f"governed_commons: {n_seeds} seeds x {n_steps} steps, "
          f"20 households acting through AI delegates\n")
    header = (f"{'condition':<22}{'influence_preserved':>20}{'exercised':>11}"
              f"{'fidelity':>10}{'stock_pct':>11}")
    print(header)
    print("-" * len(header))
    results = {}
    for name, per_seed in per_condition.items():
        ip = influence_preserved(per_seed["exercised"], baseline_exercised)
        results[name] = ip
        print(f"{name:<22}{ip:>20.3f}"
              f"{float(per_seed['exercised'].mean()):>11.3f}"
              f"{float(per_seed['influence_fidelity'].mean()):>10.3f}"
              f"{float(per_seed['stock_pct'].mean()):>11.3f}")

    print("\nundefended, the commons collapses and asks stop mattering; quota voting")
    print("alone sits on the sustainability knife-edge (outcomes track the ecology,")
    print("not the vote); add enforcement and outcomes follow the households' asks.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()
    if args.smoke:
        main(n_seeds=2, n_steps=5, seed=0)
    else:
        main(n_seeds=32, n_steps=200, seed=0)
