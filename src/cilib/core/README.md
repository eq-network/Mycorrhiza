# core — the substrate

The primitives everything composes on. One screen:

| Module | What it is |
|---|---|
| `graph.py` | `GraphState` — the immutable pytree all state lives in (`node_types`, `node_attrs`, `adj_matrices`, `edge_attrs`, `global_attrs`; dynamic/static partition for JIT) |
| `category.py` | `@transform` (reads/writes metadata) and composition: `sequential`, `parallel` (disjoint writes), `conditional`, `gated` |
| `pipeline.py` | `compile_pipeline` — execution order derived topologically from declared reads/writes; cycle and write-conflict validation. The architectural centrepiece |
| `scan.py` | pure tier: `run_scan` (`lax.scan` episode runner), `run_scan_batch` (`vmap` over seeds) |
| `time.py` | eager tier — only for genuinely effectful agents (LLM/HTTP) |
| `schedule.py` | `scheduled(mechanism, cadence, phase_offset, onset)` + `ScheduleSpec` — timing lives here, never inside mechanisms; one-shot interventions are schedules too |
| `reduce.py` | `Reducer` fused into the scan carry — O(1)-memory metrics over long runs |
| `agents.py` / `environment.py` | Agent class, Action type, Policy protocol; environment ABC |

Support modules: `initialization.py`, `property.py`, `protocols.py`, `simulation.py`.

Conventions for writing code here: `src/cilib/CLAUDE.md`.
