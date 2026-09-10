"""Write each town's game tree artifact: tree.json + runs/p-*.json.

    python -m experiments.gd_game.export [--smoke] [--variant ID] [--out DIR]

One directory per config.VARIANTS tree_id under the output root, all sharing
the tree.schema.json written beside them.

Consumes run.py's results.json (the affordable branch tree with per-path CIs)
and re-runs every path single-seed, deterministically (representative run =
``env.run(PRNGKey(seed0))``), emitting contract-v1.1 payloads through
``cilib.environments.webexport.trajectory_payload`` with the game whitelist,
the ledger_society derived series (imported from experiments/gd_bundles —
single source), mandatory rounding, and no ``adj``. The system DAG is built
WITH each branch's mechanisms, so the pipeline view shows the intervention
transform.

Everything is validated on write against schema/tree.schema.json (Python
``jsonschema`` — ``pip install -e .[export]``); eq-network CI re-validates the
same artifacts with ajv. tree.json is written last: it carries the checksums.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import tempfile

import math

import jax
import jax.random as jr
import numpy as np

from cilib.environments import ledger_society, make_env
from cilib.environments.system_graph import system_graph
from cilib.environments.webexport import trajectory_payload

from experiments.gd_bundles.derived import DERIVED

from .branches import branch_kwargs, path_id
from .config import (
    CARDS, CUT_CARDS, MAX_RUN_BYTES, MAX_TREE_BYTES, ROUND_DECIMALS, SEED0,
    VARIANTS, GameSpec, variant_spec,
)
from .run import RESULTS, SMOKE_RESULTS, smoke_spec

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema", "tree.schema.json")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "dist-tree")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _engine_info() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(__file__)).stdout.strip()
    except OSError:
        rev = "unknown"
    return {
        "package": "collective-intelligence-library",
        "version": importlib.metadata.version("collective-intelligence-library"),
        "git_rev": rev or "unknown",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
    }


def _validators():
    try:
        import jsonschema
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("tree validation needs jsonschema: "
                         "pip install -e .[export]") from e
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    def against(def_name):
        return jsonschema.Draft202012Validator(
            {"$ref": f"#/$defs/{def_name}", "$defs": schema["$defs"]})
    return {k: against(k) for k in ("tree", "run")}


def _flip_tick(payload: dict, n_humans: int):
    """First tick where more than half the HUMAN top-listen arrows point at
    AI nodes; None if never. Computed here so the page ships a fact, not a
    view-side calculation."""
    T, N = payload["meta"]["T"], payload["meta"]["N"]
    top = payload["node"]["top_listen_target"]
    for t in range(T):
        row = top[t * N: t * N + n_humans]
        if sum(1 for v in row if v >= n_humans) > n_humans / 2:
            return t
    return None


def _layout_positions(top: list, T: int, N: int, n_humans: int,
                      seed: int = 7, width: float = 1000.0,
                      height: float = 700.0) -> list:
    """Deterministic Fruchterman–Reingold layout from the wait path's human
    top-listen edges aggregated over the whole run — the canonical map every
    view and branch renders on (clusters mean 'who ended up listening to
    whom'). Aesthetic support, not a metric; positions are the one thing the
    page shows that is not itself a simulation quantity, so they are computed
    here, seeded, and shipped in the artifact."""
    edge = np.zeros((N, N))
    for t in range(T):
        row = top[t * N: t * N + n_humans]
        for i, j in enumerate(row):
            if 0 <= j < N and j != i:
                edge[i, j] += 1.0
    sym = (edge + edge.T) / max(T, 1)
    rng = np.random.default_rng(seed)
    posx = rng.uniform(60, width - 60, N)
    posy = rng.uniform(60, height - 60, N)
    ideal = math.sqrt((width - 120) * (height - 120) / N) * 0.9
    iters = 250
    for it in range(iters):
        temp = 40.0 * (1 - it / iters) + 2.0
        dx = np.zeros(N)
        dy = np.zeros(N)
        for a in range(N):
            for b in range(a + 1, N):
                ddx = posx[a] - posx[b]
                ddy = posy[a] - posy[b]
                d2 = ddx * ddx + ddy * ddy
                if d2 < 0.01:
                    ddx = rng.uniform() - 0.5
                    ddy = rng.uniform() - 0.5
                    d2 = 0.25
                d = math.sqrt(d2)
                rep = ideal * ideal / d
                dx[a] += ddx / d * rep
                dy[a] += ddy / d * rep
                dx[b] -= ddx / d * rep
                dy[b] -= ddy / d * rep
                w = sym[a, b]
                if w > 0:
                    att = d2 / ideal * min(3.0 * w, 1.5)
                    dx[a] -= ddx / d * att
                    dy[a] -= ddy / d * att
                    dx[b] += ddx / d * att
                    dy[b] += ddy / d * att
        for i in range(N):
            disp = math.hypot(dx[i], dy[i]) or 1.0
            lim = min(disp, temp)
            posx[i] = min(max(posx[i] + dx[i] / disp * lim, 40.0), width - 40.0)
            posy[i] = min(max(posy[i] + dy[i] / disp * lim, 40.0), height - 40.0)
    return [[round(float(x), 1), round(float(y), 1)]
            for x, y in zip(posx, posy)]


def export_tree(spec: GameSpec, result: dict, n_seeds: int, out_root: str,
                validators) -> None:
    out = os.path.join(out_root, spec.tree_id)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)

    sizes = {}
    trace_field_kinds = {"global": set(), "node": set(), "static": set()}
    flips: dict = {}
    wait_choices = tuple(0 for _ in spec.windows)
    layout = None
    for row in result["paths"]:
        choices = tuple(row["choices"])
        overrides, mechs = branch_kwargs(spec, choices)
        env = make_env(spec.env, mechanisms=mechs, **overrides)
        finals, trace = env.run(jr.PRNGKey(SEED0), spec.T)
        derived_all = DERIVED[spec.env](trace, env.config)
        missing = set(spec.derived) - set(derived_all)
        assert not missing, f"derived fns missing {missing}"
        payload = trajectory_payload(
            trace, finals, game_id=spec.env, n_steps=spec.T, seed=SEED0,
            params={**dataclasses.asdict(env.config),
                    "choices": list(choices)},
            scalars=env.evaluate(trace),
            system=system_graph(
                ledger_society.build_steps(env.config, list(mechs)),
                ledger_society.make_state(env.config, jr.PRNGKey(SEED0))),
            whitelist=spec.whitelist,
            derived_global={k: derived_all[k] for k in spec.derived},
            round_decimals=ROUND_DECIMALS, include_adj=False)
        validators["run"].validate(payload)
        T_, N_ = payload["meta"]["T"], payload["meta"]["N"]
        for name, arr in payload["global"].items():
            assert len(arr) == T_, f"{name} global length"
        for name, arr in payload["node"].items():
            assert len(arr) == T_ * N_, f"{name} node length"
        for kind in trace_field_kinds:
            trace_field_kinds[kind] |= set(payload[kind])

        flips[choices] = _flip_tick(payload, env.config.n_humans)
        if choices == wait_choices:
            layout = _layout_positions(
                payload["node"]["top_listen_target"], payload["meta"]["T"],
                payload["meta"]["N"], env.config.n_humans)

        rel = os.path.join("runs", path_id(choices) + ".json")
        path = os.path.join(out, rel)
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        sizes[rel] = os.path.getsize(path)
        assert sizes[rel] <= MAX_RUN_BYTES, \
            f"{rel} is {sizes[rel]} B > {MAX_RUN_BYTES} B gate"

    # --- tree.json (written last: carries the checksums) ------------------------
    assert layout is not None, "wait path missing from results"
    base_env = make_env(spec.env, **spec.overrides)
    tree = {
        "schema_version": "2",
        "contract_version": "1.1",
        "tree_id": spec.tree_id,
        "env": spec.env,
        "suite": "gradual_disempowerment",
        "created": datetime.date.today().isoformat(),
        "notes": spec.notes,
        "engine": _engine_info(),
        "config_hash": _sha256_text(_canonical({
            "base_config": dataclasses.asdict(base_env.config),
            "overrides": spec.overrides,
            "windows": list(spec.windows),
            "forced_window": [spec.forced_tick, spec.forced_card_id],
            "cards": [[c.key, c.currency, c.price] for c in CARDS],
            "T": spec.T, "n_seeds": n_seeds, "seed0": SEED0})),
        "base_config": dataclasses.asdict(base_env.config),
        "overrides": spec.overrides,
        "T": spec.T,
        "n_seeds": n_seeds,
        "seed0": SEED0,
        "representative_seed_index": 0,
        "windows": list(spec.windows),
        "cards": [dataclasses.asdict(c) for c in CARDS],
        "cut_cards": list(CUT_CARDS),
        "metrics": [{"id": m.id, "label": m.label, "direction": m.direction,
                     "kind": m.kind, "version": m.version,
                     "late_window": 0.25,
                     "definition_hash": _sha256_text(
                         inspect.getsource(base_env.metrics[m.id]))}
                    for m in spec.metrics],
        "trace_fields": {
            "global": sorted(trace_field_kinds["global"] - set(spec.derived)),
            "node": sorted(trace_field_kinds["node"]),
            "static": sorted(trace_field_kinds["static"]),
            "derived": list(spec.derived),
        },
        "forced_window": {"tick": spec.forced_tick,
                          "card_id": spec.forced_card_id},
        "layout": layout,
        "nodes": result["nodes"],
        "paths": [{**row, "id": path_id(tuple(row["choices"])),
                   "flip_tick": flips[tuple(row["choices"])]}
                  for row in result["paths"]],
        "checks": result.get("checks", []),
        "default_path": [0] * len(spec.windows),
        "checksums": {},
    }
    for rel in sorted(sizes):
        tree["checksums"][rel.replace(os.sep, "/")] = \
            _sha256_file(os.path.join(out, rel))
    validators["tree"].validate(tree)
    with open(os.path.join(out, "tree.json"), "w") as f:
        json.dump(tree, f, separators=(",", ":"))

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(out) for name in names)
    assert total <= MAX_TREE_BYTES, \
        f"{spec.tree_id} totals {total} B > {MAX_TREE_BYTES} B gate"
    biggest = max(sizes.values()) if sizes else 0
    print(f"{spec.tree_id}: {len(sizes)} path runs, biggest {biggest / 1e3:.0f} KB, "
          f"tree total {total / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", default=None, help="export only this variant_id")
    parser.add_argument("--out", default=None,
                        help=f"output root (default {DEFAULT_OUT})")
    args = parser.parse_args()

    variants = [v for v in VARIANTS if args.variant in (None, v.variant_id)]
    if not variants:
        parser.error(f"unknown variant {args.variant!r}")

    results_path = SMOKE_RESULTS if args.smoke else RESULTS
    if not os.path.exists(results_path):
        raise SystemExit(f"no {results_path} — run experiments.gd_game.run first"
                         + (" --smoke" if args.smoke else ""))
    with open(results_path) as f:
        results = json.load(f)

    out_root = args.out or (os.path.join(tempfile.gettempdir(), "gd_game_smoke")
                            if args.smoke else DEFAULT_OUT)
    validators = _validators()
    for v in variants:
        spec = variant_spec(v)
        if args.smoke:
            spec = smoke_spec(spec)
        result = results["tree"].get(spec.tree_id)
        if result is None:
            raise SystemExit(f"no {spec.tree_id} in {results_path} — run "
                             f"experiments.gd_game.run --variant {v.variant_id}")
        export_tree(spec, result, results["n_seeds"], out_root, validators)

    # the schema ships with the tree so the site validates the same document
    with open(SCHEMA_PATH) as f:
        schema_text = f.read()
    with open(os.path.join(out_root, "tree.schema.json"), "w") as f:
        f.write(schema_text)
    print(f"tree written to {out_root}")


if __name__ == "__main__":
    main()
