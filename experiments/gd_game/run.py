"""Enumerate each town's affordable branch tree -> results.json.

    python -m experiments.gd_game.run [--smoke] [--variant ID]

One tree per config.VARIANTS entry, keyed in results.json by tree_id;
``--variant`` merges over the other towns' trees rather than clobbering them.

Depth-first over the windows: each prefix of choices is priced from its
wait-completion run (future choices cannot affect the past, so the state at
window k+1 under choices c1..ck is read from the full run of c1..ck followed
by waits — which is itself a leaf of the tree, so every pricing run IS a path
result). Affordability is judged on the median across the seed batch at the
tick before the window, making the tree seed-independent.

Committed expectations (config.EXPECTATIONS, harbor only) are model
sign/ordering claims — a failure exits nonzero, reported never retuned.
config.DESIGN_GATES are the declared game-design properties (the shrinking
tray), checked for every town, whose prices and town dials are typed
tuned-for-legibility; a gate failure also exits nonzero but licenses a
revision of that town's dials, which an expectation failure does not.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile

import jax
import jax.random as jr
import numpy as np

from cilib.environments import make_env
from cilib.lab.analysis.bootstrap import bootstrap_ci

from .branches import branch_kwargs, path_id
from .config import (
    CARDS, DESIGN_GATES, EXPECTATIONS, N_SEEDS, SEED0, VARIANTS, GameSpec,
    variant_spec,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
SMOKE_RESULTS = os.path.join(tempfile.gettempdir(), "gd_game_results_smoke.json")


def smoke_spec(spec: GameSpec) -> GameSpec:
    """Tiny twin: one window, eight ticks."""
    return dataclasses.replace(spec, T=8, windows=(4,))


def _resources_at(spec: GameSpec, traces, tick: int) -> dict:
    """Median-across-seeds resources at the tick BEFORE the window opens."""
    t = max(tick - 1, 0)
    H = spec.overrides.get("n_humans", 20)
    wealth = np.asarray(traces["wealth"][:, t, :])
    share = wealth[:, :H].sum(axis=1) / np.maximum(wealth.sum(axis=1), 1e-12)
    return {
        "enforcement": float(np.median(np.asarray(traces["enforcement"][:, t]))),
        "human_wealth_share": float(np.median(share)),
    }


def _affords(card, res: dict) -> bool:
    if card.currency == "enforcement":
        return res["enforcement"] >= card.price
    return res["human_wealth_share"] >= card.price


def run_path(spec: GameSpec, choices, n_seeds: int) -> dict:
    overrides, mechs = branch_kwargs(spec, choices)
    env = make_env(spec.env, mechanisms=mechs, **overrides)
    _, traces = env.run_batch(jr.PRNGKey(SEED0), n_seeds=n_seeds,
                              n_steps=spec.T)
    values = {}
    for m in spec.metrics:
        per_seed = np.asarray(jax.vmap(env.metrics[m.id])(traces))
        point, lo, hi = bootstrap_ci(per_seed)
        values[m.id] = {"point": float(point), "lo": float(lo), "hi": float(hi)}
    resources = [_resources_at(spec, traces, w) for w in spec.windows]
    return {"values": values, "resources": resources}


def build_tree(spec: GameSpec, n_seeds: int):
    W = len(spec.windows)
    paths: dict = {}
    nodes: dict = {}

    def ensure(choices):
        if choices not in paths:
            paths[choices] = run_path(spec, choices, n_seeds)
            v = paths[choices]["values"]
            first = spec.metrics[0].id
            print(f"  {path_id(choices)}  {first}={v[first]['point']:.3f}  "
                  f"enforcement_level={v['enforcement_level']['point']:.3f}")

    def expand(prefix):
        wait_done = prefix + (0,) * (W - len(prefix))
        ensure(wait_done)
        if len(prefix) == W:
            return
        k = len(prefix)
        res = paths[wait_done]["resources"][k]
        used = set(prefix)
        afford = [c.card_id for c in CARDS
                  if c.card_id != spec.forced_card_id
                  and c.card_id not in used and _affords(c, res)]
        nodes[prefix] = {"window": k, "tick": spec.windows[k],
                         "resources": res, "affordable": afford}
        # waiting is always a legal move — recurse the wait child too, so
        # "hold now, act later" branches exist wherever later trays allow it
        for cid in [0] + afford:
            expand(prefix + (cid,))

    expand(())
    return paths, nodes


def _point(paths, choices, metric):
    row = paths.get(tuple(choices))
    if row is None:
        raise KeyError(f"path {choices} was not generated (unaffordable?)")
    return row["values"][metric]["point"]


def _check(key: str, paths, nodes) -> bool:
    if key == "levy_early_beats_wait":
        return _point(paths, (1, 0, 0), "composite") > _point(
            paths, (0, 0, 0), "composite")
    if key == "levy_early_beats_late":
        # harbor only. Window 2 is the LATEST wait-path window where the levy
        # is still affordable — by window 3 harbor's tray is empty (the
        # re-probed 2026-07-31 arc in config.CARDS), so the late comparator
        # is (0,1,0), not (0,0,1)
        return _point(paths, (1, 0, 0), "human_wealth_share") > _point(
            paths, (0, 1, 0), "human_wealth_share")
    if key == "repair_defends_rules":
        return _point(paths, (5, 0, 0), "enforcement_level") > _point(
            paths, (0, 0, 0), "enforcement_level")
    if key == "cap_defends_attention":
        return _point(paths, (3, 0, 0), "human_attention_share") > _point(
            paths, (0, 0, 0), "human_attention_share")
    if key == "tray_shrinks":
        pol = lambda prefix: sum(
            1 for cid in nodes[prefix]["affordable"]
            if next(c for c in CARDS if c.card_id == cid).currency
            == "enforcement")
        return pol((0, 0)) < pol(()) and len(nodes[()]["affordable"]) >= 1
    raise ValueError(f"unknown expectation key {key!r}")


def run_variant(v, spec: GameSpec, n_seeds: int, smoke: bool):
    """One town's tree + its checks. Returns (result, failures)."""
    paths, nodes = build_tree(spec, n_seeds)
    wait = tuple(0 for _ in spec.windows)
    for k, w in enumerate(spec.windows):
        r = paths[wait]["resources"][k]
        print(f"  wait-path window {k + 1} (tick {w}): "
              f"enforcement={r['enforcement']:.3f}, "
              f"human_wealth_share={r['human_wealth_share']:.3f}, "
              f"affordable={nodes.get(wait[:k], {}).get('affordable')}")

    checks, failed = [], []
    if not smoke:
        for group, entries in (("expectation", EXPECTATIONS.get(v.variant_id, ())),
                               ("design-gate", DESIGN_GATES)):
            for desc, key in entries:
                try:
                    passed = bool(_check(key, paths, nodes))
                except KeyError as e:
                    passed = False
                    desc = f"{desc} [{e}]"
                checks.append({"group": group, "desc": desc, "key": key,
                               "passed": passed})
                print(f"  {group} [{'PASS' if passed else 'FAIL'}]: {desc}")
                if not passed:
                    failed.append(f"{v.variant_id} {group}: {desc}")

    result = {
        "windows": list(spec.windows), "T": spec.T,
        "paths": [{"choices": list(c), **row} for c, row in sorted(paths.items())],
        "nodes": [{"prefix": list(p), **row} for p, row in sorted(nodes.items())],
        "checks": checks,
    }
    print(f"  {spec.tree_id}: {len(paths)} paths, {len(nodes)} decision nodes")
    return result, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", default=None, help="run only this variant_id")
    args = parser.parse_args()

    variants = [v for v in VARIANTS if args.variant in (None, v.variant_id)]
    if not variants:
        parser.error(f"unknown variant {args.variant!r}")
    n_seeds = 2 if args.smoke else N_SEEDS

    out, failed = {}, []
    for v in variants:
        spec = variant_spec(v)
        if args.smoke:
            spec = smoke_spec(spec)
        print(f"{v.variant_id} -> {spec.tree_id}: windows={spec.windows}, "
              f"T={spec.T}, n_seeds={n_seeds}, overrides={spec.overrides}")
        result, fails = run_variant(v, spec, n_seeds, args.smoke)
        out[spec.tree_id] = result
        failed += fails

    path = SMOKE_RESULTS if args.smoke else RESULTS
    # --variant must not clobber the other towns' trees: merge over prior
    if args.variant and os.path.exists(path):
        with open(path) as f:
            out = {**json.load(f).get("tree", {}), **out}
    with open(path, "w") as f:
        json.dump({"n_seeds": n_seeds, "seed0": SEED0, "smoke": args.smoke,
                   "tree": out}, f)
    print(f"wrote {path}  ({len(out)} trees)")
    if failed:
        print("CHECKS FAILED (expectations: report + revise, never retune; "
              "design gates: the town's dials may be revised, typed "
              "tuned-for-legibility — never the shared deck):")
        for line in failed:
            print(f"  {line}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
