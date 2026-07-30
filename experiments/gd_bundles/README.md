# gd_bundles — the GD-suite dial-lattice bundles

The playground's data source (docs/dial-lattice-design.md): one bundle per tab —
`capital-economy-knee-v1`, `influence-exchange-ampdrift-v1`,
`delegative-polity-knee-v1`, `ledger-society-channels-v1`. Grids are imported
from the WP experiment configs (single source); the ledger grid is new and
flagged for review. The browser renders these artifacts and computes nothing
(CLAUDE.md ⟦BOUNDARY⟧).

One role per file, `_template` style: `config.py` freezes specs + committed
expectations; `run.py` sweeps → per-cell bootstrap CIs → `results.json`
(expectation failures exit nonzero — report and revise the grid, never retune);
`export.py` re-runs playback cells single-seed and writes
`{manifest,scalars,runs/*}.json`, validated on write against
`schema/bundle.schema.json` (eq-network CI re-validates the same files with ajv).

    pip install -e .[export]
    python -m experiments.gd_bundles.run [--smoke]
    python -m experiments.gd_bundles.export [--smoke] \
        [--out ../eq-network/apps/site/public/lab/runs]

Regenerate whenever a suite model changes; never hand-edit a bundle (manifest
checksums make edits detectable on both sides).
