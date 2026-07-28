# CLAUDE.md — orientation for coding agents

Collective Intelligence Library ("CI Lib"): a JAX-native framework for composable
multi-agent simulation. Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it's the
pattern map. This file is the working contract.

**Alpha direction:** the five benchmark scenarios and their engine mapping live in
[docs/alpha-context.md](docs/alpha-context.md) (why/what) and
[docs/alpha-plan.md](docs/alpha-plan.md) (phasing, definitions of done); the
classical-ABM program frame (validation ladder, mechanisms-attach-via-scheduler,
influence-now vs influence-from-birth) is [docs/abm-suite-design.md](docs/abm-suite-design.md);
the economy model register (structural-robustness ensemble, assumptions cards, forking)
is [docs/model-register-design.md](docs/model-register-design.md), with the cultural
counterpart (separation × persuasion axes, spectral metrics) in
[docs/cultural-register-design.md](docs/cultural-register-design.md);
the environment boundary contract is [docs/game-boundary-design.md](docs/game-boundary-design.md)
(frozen). Public counterpart: the unlisted page at eq-network.org/lab.

## Import root

The library installs as `cilib` (distribution: `collective-intelligence-library`).
Always `from cilib.core import ...`, `from cilib.mechanisms import ...`, etc. There is
**no** top-level `core`/`engine` package anymore — that was the pre-`src/` layout.

```bash
pip install -e .        # editable; required so `import cilib` resolves
python -m pytest -q     # the safety net — keep it green
```

## Where does X go?

| You're adding… | Put it in… | Follow |
|---|---|---|
| a decision rule | `src/cilib/agents/` | `agents/README.md` |
| an atomic `state->state` step | `src/cilib/transformations/` | `transformations/README.md` |
| a composed institution (market/network/democracy) | `src/cilib/mechanisms/` | `mechanisms/README.md` |
| a runnable substrate | `src/cilib/environments/` | `environments/README.md` |
| a general-purpose in-loop readout | `src/cilib/metrics/` | — |
| a full model tied to one study/paper | `src/cilib/lab/paradigms/<name>/` | `lab/paradigms/README.md` (6-part contract) |
| paper-specific offline math (EI, causal emergence, …) | `src/cilib/lab/analysis/` | — |
| a study / sweep | `experiments/<name>/` | `experiments/_template/` |

Each catalog is a plain `REGISTRY = {...}` dict in its `__init__.py`. Adding an entry
= write the factory + add one dict line + a behavioral test. See [EXTENDING.md](EXTENDING.md).

**The lab razor:** would we merge and maintain a stranger's PR to this file the way
we'd maintain a library API? No → it goes under `cilib.lab` (research payload, no
stability promise), not a catalog. A *mechanism* is swappable into any pipeline by any
future study; a *paradigm* wires many pieces together to make one paper's argument.

## Load-bearing conventions

- **State lives in `GraphState`, not in objects.** Agents are pure factories; the
  evolving arrays live in `node_attrs` / `adj_matrices`. Don't add stateful classes.
- **Static config is closed over by transform factories, never stored in
  `global_attrs`.** `global_attrs` is static pytree aux — putting swept/per-step data
  there forces recompiles. `GraphState` carries only *evolving* arrays.
- **No data-dependent Python control flow inside transforms.** Replace `if traced:`
  with `jnp.where` / `lax.cond` (e.g. `core.category.gated`) so it traces under `lax.scan`.
- **Pure tier vs eager tier.** Pure (`core.scan`) is the default and the only tier
  that `vmap`s over seeds. Use the eager tier (`core.time`) *only* for genuinely
  effectful agents (LLM/HTTP).
- **Mechanisms/transformations declare `.reads` / `.writes`** via `@transform` so
  `compile_pipeline` can derive order. Same-family mechanisms keep disjoint writes.
- **The trajectory is the memory ceiling, not the state.** `default_trace` returning
  raw per-agent arrays costs O(T·N) — at N≥500 that is ~3000× the state itself, and
  it is what stops long or large runs (not the dense adjacency, which
  `sparse_friendship` already fixes). No metric in this repo needs the joint (T, N)
  array. Reduce the *agent* axis in `trace_fn` (a (T,) scalar series is ~8 KB and
  worth keeping), and fold anything per-agent or long-window with a
  `metrics/reducers.py` `Reducer` via `EnvSpec.run_reduced` — O(1) carry, same number
  to float32 rounding. `examples/08_streaming_metrics.py` measures the gap.
- **Simplicity is a hard requirement here.** Prefer short, inspectable code and
  deletion over new abstraction. A catalog you can read in one screen beats a clever
  registry framework.

## Verifying a change

- Behavior-preserving refactor → `python -m pytest -q` must stay green (currently 236).
- A change to a paradigm's composition → assert the new pipeline is numerically
  identical to the old one for a fixed seed before deleting the old path.
- A new catalog entry → a behavioral test asserting the *mechanism* (direction /
  ordering), not bit-exact numbers.
