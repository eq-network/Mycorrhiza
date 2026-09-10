# Experiments

In-repo studies that import `cilib`, select catalog/paradigm pieces, compile a pipeline, and
sweep it over seeds and dials. Experiments are **not** part of the installed package — they are
consumers of it. Run them from the repo root with `python -m experiments.<name>.<script>`.

**Shape (copy `_template/`):** `config.py` (frozen config + sweep axes) · `run.py`
(sweep → metrics → save) · `figures.py` (results → figures) · `README.md` (hypothesis + sweep).

| Study | What it is |
|---|---|
| `_template/` | the copyable skeleton — sweep heterogeneity, plot fit (start here) |
| `polycentric_emergence/` | full worked study: governance regimes re-described as causal emergence |
| `basin_stability/` | PDD / PRD / PLD under adversarial pressure on a resource game |
| `governed_harvest/` | earlier harvest-extraction prototype |
| `fishing_commons/` | state factory + type-contract reference |
| `assumption_audits/` | one-shot receipt probes behind the lab site's "we checked" claims (floors, defense grids, cadence scan) — deliberately not the template shape |
| `perf_baseline/` | where the time goes: compile vs run, eager vs jit, seed-batch efficiency, population scaling, per environment — the measured starting point for any engine optimisation |

See [EXTENDING.md](../EXTENDING.md) for the recipe and [ARCHITECTURE.md](../ARCHITECTURE.md) for
how experiments sit above the catalogs and paradigms.
