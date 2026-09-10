"""Marker-based extraction of the code exhibits the explainer page shows.

The page's "see the code" panels render snippets.json, never hand-pasted
strings: each snippet is cut from the engine source between two marker lines
at export time, so the exhibit is the code that actually ran, at the git rev
the manifest names. A behavioral test pins every snippet to a token it must
contain — a refactor that moves or renames the exhibit fails the export
instead of silently shipping stale prose.
"""
from __future__ import annotations

import hashlib
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# id -> (relpath, start_marker, end_marker, required_token)
# start None = start of file; end None = end of file. The snippet spans the
# first line containing start_marker up to (exclusive) the first line after it
# containing end_marker, with trailing blank lines stripped.
SNIPPETS = {
    "graphstate_def": (
        "src/cilib/core/graph.py",
        "@jax.tree_util.register_pytree_node_class", "def __post_init__",
        "node_attrs"),
    "harvest_transform": (
        "src/cilib/environments/governed_commons/dynamics.py",
        "def make_harvest", "# --- logistic regrowth",
        "@transform"),
    "regrow_transform": (
        "src/cilib/environments/governed_commons/dynamics.py",
        "def make_regrow", "# --- bookkeeping",
        "@transform"),
    "scheduled_def": (
        "src/cilib/core/schedule.py",
        "def scheduled(", "@dataclass(frozen=True)",
        "cadence"),
    "pipeline_hazards": (
        "src/cilib/core/pipeline.py",
        "def _build_dependency_graph", "def _topological_batches",
        "waw"),
    "example01": (
        "examples/01_first_transform.py", None, None,
        "run_scan"),
    "condition_attach": (
        "examples/05_export_trajectory.py",
        '"governed_commons": {', '"compute_economy": {',
        "quota_vote"),
}


def extract() -> dict:
    out = {}
    for sid, (relpath, start, end, token) in SNIPPETS.items():
        path = os.path.join(REPO_ROOT, relpath)
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        lo = 0
        if start is not None:
            lo = next(i for i, line in enumerate(lines) if start in line)
        hi = len(lines)
        if end is not None:
            hi = next(i for i in range(lo + 1, len(lines)) if end in lines[i])
        while hi > lo and not lines[hi - 1].strip():
            hi -= 1

        text = "\n".join(lines[lo:hi])
        assert token in text, f"snippet {sid}: required token {token!r} missing"
        out[sid] = {
            "path": relpath,
            "start_line": lo + 1,
            "end_line": hi,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
    return out
