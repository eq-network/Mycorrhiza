# library_explainer — fixtures for the eq-network `/library/prototype` page

The interactive explainer of the library itself (the whitepaper's arc: one
equation → GraphState → `@transform` → `scheduled()` → `compile_pipeline` →
measurement → validation discipline) renders **versioned artifacts only**, per
the CLAUDE.md ⟦BOUNDARY⟧. This experiment is the producer of that crossing.

One substrate threads the whole page: `governed_commons`, seen in turn as a
transform, a compiled pipeline, a system graph, and a trajectory.

## What it emits

| File | Consumed by (scene) | Content |
|---|---|---|
| `pipeline-subsets.json` | BatchBoard ("the compiler") | 2^4 rows over the graduated_sanctions pipeline (minus step_counter): row *k* = enabled set from *k*'s bits → hazard edges (RAW/WAR/WAW) + execution batches from `compile_pipeline`'s own ordering. The page does array lookups; no ordering logic exists in JS. |
| `system-graphs.json` | SystemGraphView ("a real pipeline") | `system_graph(build_steps, make_state)` per condition — toggling a mechanism adds/removes its node. |
| `runs/governed_commons.<condition>.json` | — unconsumed, see note | Contract-v1.1 payloads, `(T,)` globals only (whitelist `resource_level`, `policy_target`), T=200, seed 0, rounded, no `adj`, empty `node` (schema-enforced). |
| `snippets.json` | CodePanel ("see the code") | marker-extracted engine source (`snippets.py`): the exhibit is the code that ran, at the manifest's git rev, never hand-pasted prose. |
| `schedule-golden.json` | ScheduleGrid contract test | tick windows probed through the engine's own `scheduled()` wrapper — pins the page widget's predicate. |
| `state-shapes.json` | StateLayers ("one state") | the initial GraphState's field inventory (names, shapes, dtypes per family). |
| `scorecard.json` | — unconsumed, see note | the benchmark's scenario-1 reading rebuilt at export: causal `collective_influence` + descriptive metrics, mean ± se over 32 seeds, caveat class carried as data. |
| `influence-curve.json` | — unconsumed, see note | responsiveness to a one-shot ask-shift at increasing t0, per condition (16 seeds). Measured shape: defended responds at every t0 (~+0.49); undefended never meaningfully responds — see `build_influence_curve`'s docstring. |
| `manifest.json` | fixture loader + CodePanel footer | engine provenance (version, git rev), T, seed, conditions, sha256 checksums. Written last. |
| `explainer.schema.json` | eq-network CI (ajv) | copy of `schema/explainer.schema.json` so the site validates the same document. |

**Unconsumed fixtures (2026-08-08 page revision, owner direction):** the page's
measurement segment and organising-frame interlude were removed and replaced by
a categorical-diagrams segment that renders `pipeline-subsets.json` only, so
`scorecard.json`, `influence-curve.json`, and `runs/*.json` currently have no
consuming scene. They stay in the export deliberately — contract v1.1 is
unchanged, eq-network still schema-validates them at build time, and stripping
them would be a versioned contract change on both sides for ~25 KB. Revisit if
the contract is bumped for another reason.

Committed expectations (gd_bundles-style) run on the full export only: the
ordering claims the page narrates must hold in the shipped fixtures or the
export fails — a miss is reported, never retuned. The influence-curve
expectation was corrected once (2026-08-07, first full run): the original
guess assumed gradual decay of undefended influence; this substrate collapses
fast, so undefended influence is ~zero at every t0. Numbers untouched.

Size gates: ≤100 KB per file, ≤400 KB total — enforced on write and re-checked
by both validators.

## Regenerate + paste

```bash
python -m experiments.library_explainer.export            # -> dist-fixtures/
python -m experiments.library_explainer.export --smoke    # tiny run for tests
```

Copy `dist-fixtures/*` to `eq-network/apps/site/src/data/library-explainer/`
— pasted, never hand-edited. eq-network CI re-validates with ajv
(`apps/site/scripts/validate-explainer-fixtures.mjs`); the two independent
validators of one shared schema are the drift guard (same pattern as
`experiments/gd_bundles`).

