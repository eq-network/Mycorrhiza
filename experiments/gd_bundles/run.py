"""Sweep every BundleSpec's lattice -> results.json (per-cell bootstrap CIs).

    python -m experiments.gd_bundles.run [--smoke] [--bundle ID]

Per cell: run_batch over the shared seed batch, vmap each env metric over the
seed axis (the experiments/benchmark/harness.py pattern), bootstrap CIs.
Committed expectations (config.EXPECTATIONS) are evaluated on the finished
lattice, RECORDED into results.json, and any failure exits nonzero — report
and revise the grid, never retune (WP1 §5 discipline).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import tempfile

import jax
import jax.random as jr
import numpy as np

from cilib.environments import make_env
from cilib.lab.analysis.bootstrap import bootstrap_ci

from .config import BUNDLES, EXPECTATIONS, N_SEEDS, SEED0, T, BundleSpec

RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
SMOKE_RESULTS = os.path.join(tempfile.gettempdir(), "gd_bundles_results_smoke.json")


def smoke_spec(spec: BundleSpec) -> BundleSpec:
    """Tiny twin of a spec: first two values per axis, playback first cell."""
    axes = tuple(dataclasses.replace(a, values=a.values[:2]) for a in spec.axes)
    return dataclasses.replace(spec, axes=axes, playback=(tuple(0 for _ in axes),))


def cell_indices(spec: BundleSpec):
    return itertools.product(*(range(len(a.values)) for a in spec.axes))


def cell_overrides(spec: BundleSpec, cell):
    return {a.name: a.values[i] for a, i in zip(spec.axes, cell)}


def run_bundle(spec: BundleSpec, n_seeds: int, n_steps: int) -> dict:
    shape = [len(a.values) for a in spec.axes]
    rows = []
    for cell in cell_indices(spec):
        env = make_env(spec.env, **spec.overrides, **cell_overrides(spec, cell))
        _, traces = env.run_batch(jr.PRNGKey(SEED0), n_seeds=n_seeds, n_steps=n_steps)
        values = {}
        for m in spec.metrics:
            per_seed = np.asarray(jax.vmap(env.metrics[m.id])(traces))
            point, lo, hi = bootstrap_ci(per_seed)
            values[m.id] = {"point": float(point), "lo": float(lo), "hi": float(hi)}
        rows.append({"cell": list(cell), "values": values})
        print(f"  {spec.bundle_id} {list(cell)}  "
              + "  ".join(f"{m.id}={values[m.id]['point']:.3f}" for m in spec.metrics[:3]))
    return {"shape": shape, "cells": rows}


def _point(result, cell, metric):
    for row in result["cells"]:
        if tuple(row["cell"]) == tuple(cell):
            return row["values"][metric]["point"]
    raise KeyError(cell)


# committed expectation checks, keyed from config.EXPECTATIONS. Third-axis
# index 0 is each defense/floor dial's off/default value. v2 grids: economy
# rows are (efficiency, recycle, ownership) with recycle index 4 = r 1.0;
# influence drift index 3 = u 0.08 on the restored full grid.
def _check(key: str, result: dict) -> bool:
    if key == "capital_three_regimes":
        h = [_point(result, (i, 4, 0), "human_sector_share") for i in (0, 2, 7)]
        return h[0] > 0.9 and 0.1 < h[1] < 0.9 and h[2] < 0.1
    if key == "capital_ai_wealth_orders":
        return (_point(result, (0, 4, 0), "ai_wealth_share") < 0.2
                and _point(result, (7, 4, 0), "ai_wealth_share") > 0.5)
    if key == "capital_recycle_minimum":
        mid = _point(result, (2, 1, 0), "output_late")
        return (mid < _point(result, (2, 0, 0), "output_late")
                and mid < _point(result, (2, 2, 0), "output_late"))
    if key == "influence_amp_captures":
        return (_point(result, (0, 3, 0), "human_influence_share")
                - _point(result, (5, 3, 0), "human_influence_share") > 0.10)
    if key == "influence_frozen_protects":
        return _point(result, (5, 0, 0), "human_influence_share") > 0.7
    if key == "influence_lambda_belief_capture":
        return (_point(result, (5, 3, 3), "consensus_error")
                > _point(result, (5, 3, 0), "consensus_error"))
    if key == "polity_both_regimes":
        return (_point(result, (0, 1, 0), "human_power_share") > 0.75
                and _point(result, (6, 0, 0), "human_power_share") < 0.5)
    if key == "polity_churn_defends":
        return _point(result, (6, 3, 0), "human_power_share") > 0.6
    if key == "ledger_coupling_costs":
        return (_point(result, (0, 0, 0), "composite")
                > _point(result, (4, 4, 4), "composite"))
    if key == "ledger_regime_knee":
        return (_point(result, (3, 3, 0), "enforcement_level")
                - _point(result, (3, 3, 4), "enforcement_level") > 0.5)
    raise ValueError(f"unknown expectation key {key!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bundle", default=None, help="run only this bundle_id")
    args = parser.parse_args()

    specs = [s for s in BUNDLES if args.bundle in (None, s.bundle_id)]
    if not specs:
        parser.error(f"unknown bundle {args.bundle!r}")
    n_seeds = 2 if args.smoke else N_SEEDS

    out, failed = {}, []
    for spec in specs:
        spec_run = smoke_spec(spec) if args.smoke else spec
        n_steps = 8 if args.smoke else spec.T
        print(f"{spec.bundle_id}: {np.prod([len(a.values) for a in spec_run.axes])} cells, T={n_steps}")
        result = run_bundle(spec_run, n_seeds, n_steps)
        result["T"] = n_steps
        result["expectations"] = []
        if not args.smoke:
            for desc, key in EXPECTATIONS.get(spec.bundle_id, []):
                passed = bool(_check(key, result))
                result["expectations"].append({"desc": desc, "key": key, "passed": passed})
                print(f"  expectation [{'PASS' if passed else 'FAIL'}]: {desc}")
                if not passed:
                    failed.append(f"{spec.bundle_id}: {desc}")
        out[spec.bundle_id] = result

    path = SMOKE_RESULTS if args.smoke else RESULTS
    # --bundle must not clobber the other bundles' lattices: merge over prior
    if args.bundle and os.path.exists(path):
        with open(path) as f:
            out = {**json.load(f).get("bundles", {}), **out}
    with open(path, "w") as f:
        json.dump({"n_seeds": n_seeds, "seed0": SEED0,
                   "smoke": args.smoke, "bundles": out}, f)
    print(f"wrote {path}")
    if failed:
        print("COMMITTED EXPECTATIONS FAILED (report + revise grid, never retune):")
        for line in failed:
            print(f"  {line}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
