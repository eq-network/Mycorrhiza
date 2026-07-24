"""Export one rollout as web-playground JSON (docs/web-trajectory-contract.md).

The lab page's playground consumes a flat trajectory schema that any producer —
the in-browser JS port, a future JAX endpoint, or this script — emits
identically. This is the file-producer: it runs a registered environment once
and maps the trace onto the contract mechanically (no per-environment code):

    (T,) fields -> "global";  (T, N) varying -> "node";  (T, N) constant -> "static"

Used for parity fixtures against the browser port and, later, as StaticEngine
payloads for published-run replays. Not a study (no sweep, no CIs) — for that,
see experiments/benchmark/.

    python examples/05_export_trajectory.py --env governed_commons --T 500
    python examples/05_export_trajectory.py --env compute_economy \
        --condition tax_and_ownership_cap --T 500 --out econ.json
"""
import argparse
import dataclasses
import json
import os
import tempfile

import jax.random as jr
import numpy as np

from cilib.core.schedule import ScheduleSpec, apply_schedule
from cilib.environments import make_env
from cilib.mechanisms import (
    REGISTRY as MECHANISMS,
    AIRevenueTaxConfig, OwnershipCapConfig, QuotaVoteConfig, SanctionConfig,
)

# Kept in sync by hand with experiments/benchmark/scenarios.py (same convention
# as examples/04): this script depends only on the installed package.
CONDITIONS = {
    "governed_commons": {
        "baseline": [],
        "quota_voting": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5))],
        "graduated_sanctions": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
                                ("graduated_sanction", SanctionConfig(), None)],
    },
    "compute_economy": {
        "baseline": [],
        "ai_revenue_tax": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                            ScheduleSpec(onset=50))],
        "tax_and_ownership_cap": [("ai_revenue_tax", AIRevenueTaxConfig(tax_rate=0.5),
                                   ScheduleSpec(onset=50)),
                                  ("ownership_cap", OwnershipCapConfig(cap_share=0.35), None)],
    },
}


def export(env_name: str, condition: str, n_steps: int, seed: int) -> dict:
    mechs = tuple(apply_schedule(MECHANISMS[key](cfg), sched)
                  for key, cfg, sched in CONDITIONS[env_name][condition])
    env = make_env(env_name, mechanisms=mechs)
    _, trace = env.run(jr.PRNGKey(seed), n_steps)

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
                payload["node"][name] = arr.ravel().tolist()   # row-major t*N+i
        else:
            raise ValueError(f"trace field {name!r} has unsupported shape {arr.shape}")

    payload["meta"] = {
        "gameId": env_name,
        "T": n_steps,
        "N": n_agents,
        "seed": seed,
        "params": {**dataclasses.asdict(env.config), "condition": condition},
        "scalars": {k: float(v) for k, v in env.evaluate(trace).items()},
    }
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="governed_commons", choices=sorted(CONDITIONS))
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output path (default: <env>_<condition>_s<seed>.json)")
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()

    if args.condition not in CONDITIONS[args.env]:
        parser.error(f"--condition must be one of {sorted(CONDITIONS[args.env])} for {args.env}")
    if args.smoke:
        args.T = 5
        args.out = args.out or os.path.join(tempfile.gettempdir(),
                                            "cilib_export_smoke.json")

    payload = export(args.env, args.condition, args.T, args.seed)
    out = args.out or f"{args.env}_{args.condition}_s{args.seed}.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    scalars = "  ".join(f"{k}={v:.3f}" for k, v in payload["meta"]["scalars"].items())
    print(f"wrote {out}  (T={args.T}, N={payload['meta']['N']})\n{scalars}")
